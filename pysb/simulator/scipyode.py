from pysb.simulator.base import Simulator, SimulatorException, SimulationResult
import scipy.integrate
try:
    # weave is not available under Python 3.
    from scipy.weave import inline as weave_inline
    import scipy.weave.build_tools
except ImportError:
    weave_inline = None
import distutils
import pysb.bng
import sympy
import re
import numpy as np
import warnings


def _exec(code, locals):
    # This is function call under Python 3, and a statement with a
    # tuple under Python 2. The effect should be the same.
    exec(code, locals)


class ScipyOdeSimulator(Simulator):
    # some sane default options for a few well-known integrators
    default_integrator_options = {
        'vode': {
            'method': 'bdf',
            'with_jacobian': True,
            # Set nsteps as high as possible to give our users flexibility in
            # choosing their time step. (Let's be safe and assume vode was
            # compiled
            # with 32-bit ints. What would actually happen if it was and we
            # passed
            # 2**64-1 though?)
            'nsteps': 2 ** 31 - 1,
        },
        'cvode': {
            'method': 'bdf',
            'iteration': 'newton',
        },
        'lsoda': {
            'mxstep': 2**31-1,
        }
    }

    def __init__(self, model, tspan=None, initials=None, param_values=None,
                 verbose=False, **kwargs):
        super(ScipyOdeSimulator, self).__init__(model,
                                                tspan=tspan,
                                                initials=initials,
                                                param_values=param_values,
                                                verbose=verbose,
                                                **kwargs)
        # We'll need to know if we're using the Jacobian when we get to run()
        self._use_analytic_jacobian = kwargs.get('use_analytic_jacobian',
                                                 False)
        self.cleanup = kwargs.get('cleanup', True)
        integrator = kwargs.get('integrator', 'vode')
        # Generate the equations for the model
        pysb.bng.generate_equations(self._model, self.cleanup, self.verbose)

        def eqn_substitutions(eqns):
            """String substitutions on the sympy C code for the ODE RHS and
            Jacobian functions to use appropriate terms for variables and
            parameters."""
            # Substitute expanded parameter formulas for any named expressions
            for e in self._model.expressions:
                eqns = re.sub(r'\b(%s)\b' % e.name, '(' + sympy.ccode(
                    e.expand_expr()) + ')', eqns)

            # Substitute sums of observable species that could've been added
            # by expressions
            for obs in self._model.observables:
                obs_string = ''
                for i in range(len(obs.coefficients)):
                    if i > 0:
                        obs_string += "+"
                    if obs.coefficients[i] > 1:
                        obs_string += str(obs.coefficients[i]) + "*"
                    obs_string += "__s" + str(obs.species[i])
                if len(obs.coefficients) > 1:
                    obs_string = '(' + obs_string + ')'
                eqns = re.sub(r'\b(%s)\b' % obs.name, obs_string, eqns)

            # Substitute 'y[i]' for 'si'
            eqns = re.sub(r'\b__s(\d+)\b',
                          lambda m: 'y[%s]' % (int(m.group(1))),
                          eqns)

            # Substitute 'p[i]' for any named parameters
            for i, p in enumerate(self._model.parameters):
                eqns = re.sub(r'\b(%s)\b' % p.name, 'p[%d]' % i, eqns)
            return eqns

        # ODE RHS -----------------------------------------------
        # Prepare the string representations of the RHS equations
        code_eqs = '\n'.join(['ydot[%d] = %s;' %
                              (i, sympy.ccode(self._model.odes[i]))
                              for i in range(len(self._model.odes))])
        code_eqs = eqn_substitutions(code_eqs)

        self._test_inline()

        # If we can't use weave.inline to run the C code, compile it as
        # Python code instead for use with
        # exec. Note: C code with array indexing, basic math operations,
        # and pow() just happens to also
        # be valid Python.  If the equations ever have more complex things
        # in them, this might fail.
        if not self._use_inline:
            code_eqs_py = compile(code_eqs, '<%s odes>' % self._model.name,
                                  'exec')
        else:
            for arr_name in ('ydot', 'y', 'p'):
                macro = arr_name.upper() + '1'
                code_eqs = re.sub(r'\b%s\[(\d+)\]' % arr_name,
                                  '%s(\\1)' % macro, code_eqs)

        def rhs(t, y, p):
            ydot = self.ydot
            # note that the evaluated code sets ydot as a side effect
            if self._use_inline:
                weave_inline(code_eqs, ['ydot', 't', 'y', 'p'])
            else:
                _exec(code_eqs_py, locals())
            return ydot

        # JACOBIAN -----------------------------------------------
        # We'll keep the code for putting together the matrix in Sympy
        # in case we want to do manipulations of the matrix later (e.g., to
        # put together the sensitivity matrix)
        jac_fn = None
        if self._use_analytic_jacobian:
            species_names = ['__s%d' % i for i in
                             range(len(self._model.species))]
            jac_matrix = []
            # Rows of jac_matrix are by equation f_i:
            # [[df1/x1, df1/x2, ..., df1/xn],
            #  [   ...                     ],
            #  [dfn/x1, dfn/x2, ..., dfn/xn],
            for eqn in self._model.odes:
                # Derivatives for f_i...
                jac_row = []
                for species_name in species_names:
                    # ... with respect to s_j
                    d = sympy.diff(eqn, species_name)
                    jac_row.append(d)
                jac_matrix.append(jac_row)

            # Next, prepare the stringified Jacobian equations
            jac_eqs_list = []
            for i, row in enumerate(jac_matrix):
                for j, entry in enumerate(row):
                    # Skip zero entries in the Jacobian
                    if entry == 0:
                        continue
                    jac_eq_str = 'jac[%d, %d] = %s;' % (
                    i, j, sympy.ccode(entry))
                    jac_eqs_list.append(jac_eq_str)
            jac_eqs = eqn_substitutions('\n'.join(jac_eqs_list))

            # Try to inline the Jacobian if possible (as above for RHS)
            if not self._use_inline:
                jac_eqs_py = compile(jac_eqs,
                                     '<%s jacobian>' % self._model.name, 'exec')
            else:
                # Substitute array refs with calls to the JAC1 macro for inline
                jac_eqs = re.sub(r'\bjac\[(\d+), (\d+)\]',
                                 r'JAC2(\1, \2)', jac_eqs)
                # Substitute calls to the Y1 and P1 macros
                for arr_name in ('y', 'p'):
                    macro = arr_name.upper() + '1'
                    jac_eqs = re.sub(r'\b%s\[(\d+)\]' % arr_name,
                                     '%s(\\1)' % macro, jac_eqs)

            def jacobian(t, y, p):
                jac = self.jac
                # note that the evaluated code sets jac as a side effect
                if self._use_inline:
                    weave_inline(jac_eqs, ['jac', 't', 'y', 'p']);
                else:
                    _exec(jac_eqs_py, locals())
                return jac

            # Initialize the jacobian argument to None if we're not going to
            #  use it
            # jac = self.jac as defined in jacobian() earlier
            # Initialization of matrix for storing the Jacobian
            self.jac = np.zeros(
                (len(self._model.odes), len(self._model.species)))
            jac_fn = jacobian

        # build integrator options list from our defaults and any kwargs
        # passed to this function
        options = {}
        if self.default_integrator_options.get(integrator):
            options.update(
                self.default_integrator_options[integrator])  # default options

        options.update(kwargs.get('integrator_options', {}))  # overwrite
        # defaults
        self.opts = options
        self.ydot = np.ndarray(len(self._model.species))

        # Integrator
        if integrator == 'lsoda':
            # lsoda is accessed via scipy.integrate.odeint which,
            # as a function,
            # requires that we pass its args at the point of call. Thus we need
            # to stash stuff like the rhs and jacobian functions in self so we
            # can pass them in later.
            self.integrator = integrator
            # lsoda's rhs and jacobian function arguments are in a different
            # order to other integrators, so we define these shims that swizzle
            # the argument order appropriately.
            self.func = lambda t, y, p: rhs(y, t, p)
            if jac_fn is None:
                self.jac_fn = None
            else:
                self.jac_fn = lambda t, y, p: jac_fn(y, t, p)
        else:
            # The scipy.integrate.ode integrators on the other hand are object
            # oriented and hold the functions and such internally. Once we set
            # up the integrator object we only need to retain a reference to it
            # and can forget about the other bits.
            self.integrator = scipy.integrate.ode(rhs, jac=jac_fn)
            with warnings.catch_warnings():
                warnings.filterwarnings('error', 'No integrator name match')
                self.integrator.set_integrator(integrator, **options)

    @classmethod
    def _test_inline(cls):
        """Detect whether scipy.weave.inline is functional."""
        if not hasattr(cls, '_use_inline'):
            cls._use_inline = False
            try:
                if weave_inline is not None:
                    weave_inline('int i=0; i=i;', force=1)
                    cls._use_inline = True
            except (scipy.weave.build_tools.CompileError,
                    distutils.errors.CompileError, ImportError):
                pass

    def run(self, tspan=None, param_values=None, initials=None):
        if tspan is not None:
            self.tspan = tspan
        if self.tspan is None:
            raise SimulatorException("tspan must be defined before "
                                     "simulation can run")
        trajectories = np.ndarray((1, len(self.tspan),
                                  len(self._model.species)))
        if param_values is not None:
            self.param_values = param_values
        if initials is not None:
            self.initials = initials
        y0 = self.initials_list
        param_values = self.param_values
        if self.integrator == 'lsoda':
            trajectories[0] = scipy.integrate.odeint(self.func,
                                                y0,
                                                self.tspan,
                                                Dfun=self.jac_fn,
                                                args=(param_values,),
                                                **self.opts)
        else:
            # perform the actual integration
            self.integrator.set_initial_value(y0, self.tspan[0])
            # Set parameter vectors
            # for RHS func and Jacobian
            self.integrator.set_f_params(param_values)
            if self._use_analytic_jacobian:
                self.integrator.set_jac_params(param_values)
            trajectories[0][0] = y0
            i = 1
            if self.verbose:
                print("Integrating...")
                print("\tTime")
                print("\t----")
                print("\t%g" % self.integrator.t)
            while self.integrator.successful() and self.integrator.t < \
                    self.tspan[-1]:
                trajectories[0][i] = self.integrator.integrate(self.tspan[i])
                i += 1
                if self.verbose: print("\t%g" % self.integrator.t)
            if self.verbose: print("...Done.")
            if self.integrator.t < self.tspan[-1]:
                trajectories[0, i:, :] = 'nan'
        self.tout = [self.tspan, ]
        return SimulationResult(self, trajectories)
