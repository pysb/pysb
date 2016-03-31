import pysb.core
import pysb.bng
import numpy
import scipy.integrate
import code
try:
    # weave is not available under Python 3.
    from scipy.weave import inline as weave_inline
    import scipy.weave.build_tools
except ImportError:
    weave_inline = None

import distutils.errors
import sympy
import re
import itertools
import warnings
try:
    from future_builtins import zip
except ImportError:
    pass

def _exec(code, locals):
    # This is function call under Python 3, and a statement with a
    # tuple under Python 2. The effect should be the same.
    exec(code, locals)

# some sane default options for a few well-known integrators
default_integrator_options = {
    'vode': {
        'method': 'bdf',
        'with_jacobian': True,
        # Set nsteps as high as possible to give our users flexibility in
        # choosing their time step. (Let's be safe and assume vode was compiled
        # with 32-bit ints. What would actually happen if it was and we passed
        # 2**64-1 though?)
        'nsteps': 2**31 - 1,
        },
    'cvode': {
        'method': 'bdf',
        'iteration': 'newton',
        },
    }

class Solver(object):
    """An interface for numeric integration of models.

    Parameters
    ----------
    model : pysb.Model
        Model to integrate.
    tspan : vector-like
        Time values over which to integrate. The first and last values define
        the time range, and the returned trajectories will be sampled at every
        value.
    use_analytic_jacobian : boolean, optional
        Whether to provide the solver a Jacobian matrix derived analytically
        from the model ODEs. Defaults to False. If False, the integrator may
        approximate the Jacobian by finite-differences calculations when
        necessary (depending on the integrator and settings).
    integrator : string, optional (default: 'vode')
        Name of the integrator to use, taken from the list of integrators known
        to :py:class:`scipy.integrate.ode`.
    cleanup : bool, optional
        If True (default), delete the temporary files after the simulation is
        finished. If False, leave them in place. Useful for debugging.
    verbose : bool, optional (default: False)
        Verbose output 
    integrator_options
        Additional parameters for the integrator.

    Attributes
    ----------
    model : pysb.Model
        Model passed to the constructor
    tspan : vector-like
        Time values passed to the constructor.
    y : numpy.ndarray
        Species trajectories. Dimensionality is ``(len(tspan),
        len(model.species))``.
    yobs : numpy.ndarray with record-style data-type
        Observable trajectories. Length is ``len(tspan)`` and record names
        follow ``model.observables`` names.
    yobs_view : numpy.ndarray
        An array view (sharing the same data buffer) on ``yobs``. Dimensionality
        is ``(len(tspan), len(model.observables))``.
    yexpr : numpy.ndarray with record-style data-type
        Expression trajectories. Length is ``len(tspan)`` and record names
        follow ``model.expressions_dynamic()`` names.
    yexpr_view : numpy.ndarray
        An array view (sharing the same data buffer) on ``yexpr``. Dimensionality
        is ``(len(tspan), len(model.expressions_dynamic()))``.
    integrator : scipy.integrate.ode
        Integrator object.

    Notes
    -----
    The expensive step of generating the code for the right-hand side of the
    model's ODEs is performed during initialization. If you need to integrate
    the same model repeatedly with different parameters then you should build a
    single Solver object and then call its ``run`` method as needed.

    """

    @staticmethod
    def _test_inline():
        """Detect whether scipy.weave.inline is functional."""
        if not hasattr(Solver, '_use_inline'):
            Solver._use_inline = False
            try:
                if weave_inline is not None:
                    weave_inline('int i=0; i=i;', force=1)
                    Solver._use_inline = True
            except (scipy.weave.build_tools.CompileError,
                    distutils.errors.CompileError, ImportError):
                pass


    def __init__(self, model, tspan, use_analytic_jacobian=False,
                 integrator='vode', cleanup=True,
                 verbose=False, **integrator_options):

        self.verbose = verbose
        self.model = model
        self.tspan = tspan
        # We'll need to know if we're using the Jacobian when we get to run()
        self._use_analytic_jacobian = use_analytic_jacobian        
        # Generate the equations for the model
        pysb.bng.generate_equations(self.model, cleanup, self.verbose)

        def eqn_substitutions(eqns):
            """String substitutions on the sympy C code for the ODE RHS and
            Jacobian functions to use appropriate terms for variables and
            parameters."""
            # Substitute expanded parameter formulas for any named expressions
            for e in self.model.expressions:
                eqns = re.sub(r'\b(%s)\b' % e.name, '('+sympy.ccode(
                    e.expand_expr())+')', eqns)

            # Substitute sums of observable species that could've been added
            # by expressions
            for obs in self.model.observables:
                obs_string = ''
                for i in range(len(obs.coefficients)):
                    if i > 0:
                        obs_string += "+"
                    if obs.coefficients[i] > 1:
                        obs_string += str(obs.coefficients[i])+"*"
                    obs_string += "__s"+str(obs.species[i])
                if len(obs.coefficients) > 1:
                    obs_string = '(' + obs_string + ')'
                eqns = re.sub(r'\b(%s)\b' % obs.name, obs_string, eqns)

            # Substitute 'y[i]' for 'si'
            eqns = re.sub(r'\b__s(\d+)\b', lambda m: 'y[%s]' % (int(m.group(1))),
                       eqns)

            # Substitute 'p[i]' for any named parameters
            for i, p in enumerate(self.model.parameters):
                eqns = re.sub(r'\b(%s)\b' % p.name, 'p[%d]' % i, eqns)
            return eqns

        # ODE RHS -----------------------------------------------
        # Prepare the string representations of the RHS equations
        code_eqs = '\n'.join(['ydot[%d] = %s;' %
                              (i, sympy.ccode(self.model.odes[i]))
                              for i in range(len(self.model.odes))])
        code_eqs = eqn_substitutions(code_eqs)
        
        Solver._test_inline()

        # If we can't use weave.inline to run the C code, compile it as Python code instead for use with
        # exec. Note: C code with array indexing, basic math operations, and pow() just happens to also
        # be valid Python.  If the equations ever have more complex things in them, this might fail.
        if not Solver._use_inline:
            code_eqs_py = compile(code_eqs, '<%s odes>' % self.model.name, 'exec')
        else:
            for arr_name in ('ydot', 'y', 'p'):
                macro = arr_name.upper() + '1'
                code_eqs = re.sub(r'\b%s\[(\d+)\]' % arr_name,'%s(\\1)' % macro, code_eqs)

        def rhs(t, y, p):
            ydot = self.ydot
            # note that the evaluated code sets ydot as a side effect
            if Solver._use_inline:
                weave_inline(code_eqs, ['ydot', 't', 'y', 'p']);
            else:
                _exec(code_eqs_py, locals())
            return ydot
        
        # JACOBIAN -----------------------------------------------
        # We'll keep the code for putting together the matrix in Sympy
        # in case we want to do manipulations of the matrix later (e.g., to
        # put together the sensitivity matrix)
        jac_fn = None
        if self._use_analytic_jacobian:
            species_names = ['__s%d' % i for i in range(len(self.model.species))]
            jac_matrix = []
            # Rows of jac_matrix are by equation f_i:
            # [[df1/x1, df1/x2, ..., df1/xn],
            #  [   ...                     ],
            #  [dfn/x1, dfn/x2, ..., dfn/xn],
            for eqn in self.model.odes:
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
                    jac_eq_str = 'jac[%d, %d] = %s;' % (i, j, sympy.ccode(entry))
                    jac_eqs_list.append(jac_eq_str)
            jac_eqs = eqn_substitutions('\n'.join(jac_eqs_list))
            
            # Try to inline the Jacobian if possible (as above for RHS)
            if not Solver._use_inline:
                jac_eqs_py = compile(jac_eqs, '<%s jacobian>' % self.model.name, 'exec')
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
                if Solver._use_inline:
                    weave_inline(jac_eqs, ['jac', 't', 'y', 'p']);
                else:
                    _exec(jac_eqs_py, locals())
                return jac
            
            # Initialize the jacobian argument to None if we're not going to use it
            # jac = self.jac as defined in jacobian() earlier
            # Initialization of matrix for storing the Jacobian
            self.jac = numpy.zeros((len(self.model.odes), len(self.model.species)))
            jac_fn = jacobian

        # build integrator options list from our defaults and any kwargs passed to this function
        options = {}
        if default_integrator_options.get(integrator):
            options.update(default_integrator_options[integrator]) # default options

        options.update(integrator_options) # overwrite defaults
        self.opts = options
        self.y = numpy.ndarray((len(self.tspan), len(self.model.species))) # species concentrations
        self.ydot = numpy.ndarray(len(self.model.species))

        # Initialize record array for observable timecourses
        if len(self.model.observables):
            self.yobs = numpy.ndarray(len(self.tspan),
                                      list(zip(self.model.observables.keys(),
                                          itertools.repeat(float))))
        else:
            self.yobs = numpy.ndarray((len(self.tspan), 0))
        self.yobs_view = self.yobs.view(float).reshape(len(self.yobs), -1)

        # Initialize array for expression timecourses
        exprs = self.model.expressions_dynamic()
        if len(exprs):
            self.yexpr = numpy.ndarray(len(self.tspan), list(zip(exprs.keys(),
                                                       itertools.repeat(
                                                           float))))
        else:
            self.yexpr = numpy.ndarray((len(self.tspan), 0))
        self.yexpr_view = self.yexpr.view(float).reshape(len(self.yexpr), -1)

        # Integrator
        if integrator == 'lsoda':
            # lsoda is accessed via scipy.integrate.odeint which, as a function,
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


    def run(self, param_values=None, y0=None):
        """Perform an integration.

        Returns nothing; access the Solver object's ``y``, ``yobs``, or
        ``yobs_view`` attributes to retrieve the results.

        Parameters
        ----------
        param_values : vector-like or dictionary, optional
            Values to use for every parameter in the model. Ordering is
            determined by the order of model.parameters. 
            If passed as a dictionary, keys must be parameter names.
            If not specified, parameter values will be taken directly from
            model.parameters.
        y0 : vector-like, optional
            Values to use for the initial condition of all species. Ordering is
            determined by the order of model.species. If not specified, initial
            conditions will be taken from model.initial_conditions (with initial
            condition parameter values taken from `param_values` if specified).
        """

        if param_values is not None and not isinstance(param_values, dict):
            # accept vector of parameter values as an argument
            if len(param_values) != len(self.model.parameters):
                raise ValueError("param_values must be the same length as "
                                 "model.parameters")
            if not isinstance(param_values, numpy.ndarray):
                param_values = numpy.array(param_values)
        else:
            # create parameter vector from the values in the model
            param_values_dict = param_values if isinstance(param_values, dict) else {}
            param_values = numpy.array([p.value for p in self.model.parameters])
            for key in param_values_dict.keys():
                try:
                    pi = self.model.parameters.index(self.model.parameters[key])
                except KeyError:
                    raise IndexError("param_values dictionary has unknown "
                                     "parameter name (%s)" % key)
                param_values[pi] = param_values_dict[key]

        # The substitution dict must have Symbols as keys, not strings
        subs = dict((p, param_values[i]) for i, p in
                    enumerate(self.model.parameters))

        if y0 is not None:
            # check if y0 is a dict (not supported yet)
            if isinstance(y0, dict):
                raise NotImplementedError
            # accept vector of species amounts as an argument
            if len(y0) != self.y.shape[1]:
                raise ValueError("y0 must be the same length as model.species")
            if not isinstance(y0, numpy.ndarray):
                y0 = numpy.array(y0)
        else:
            y0 = numpy.zeros((self.y.shape[1],))

            for cp, value_obj in self.model.initial_conditions:
                if value_obj in self.model.parameters:
                    pi = self.model.parameters.index(value_obj)
                    value = param_values[pi]
                elif value_obj in self.model.expressions:
                    value = value_obj.expand_expr().evalf(subs=subs)
                else:
                    raise ValueError("Unexpected initial condition value type")
                si = self.model.get_species_index(cp)
                if si is None:
                    raise IndexError("Species not found in model: %s" %
                                     repr(cp))
                y0[si] = value
        
        if self.integrator == 'lsoda':
            self.y = scipy.integrate.odeint(self.func, y0, self.tspan,
                                            Dfun=self.jac_fn,
                                            args=(param_values,), **self.opts)
        else:
            # perform the actual integration
            self.integrator.set_initial_value(y0, self.tspan[0])
            # Set parameter vectors for RHS func and Jacobian
            self.integrator.set_f_params(param_values)
            if self._use_analytic_jacobian:
                self.integrator.set_jac_params(param_values)
            self.y[0] = y0
            i = 1
            if self.verbose:
                print("Integrating...")
                print("\tTime")
                print("\t----")
                print("\t%g" % self.integrator.t)
            while self.integrator.successful() and self.integrator.t < self.tspan[-1]:
                self.y[i] = self.integrator.integrate(self.tspan[i]) # integration
                i += 1
                ######
    #             self.integrator.integrate(self.tspan[i],step=True)
    #             if self.integrator.t >= self.tspan[i]: i += 1
                ######
                if self.verbose: print("\t%g" % self.integrator.t)
            if self.verbose: print("...Done.")
            if self.integrator.t < self.tspan[-1]:
                self.y[i:, :] = 'nan'

        # calculate observables
        for i, obs in enumerate(self.model.observables):
            self.yobs_view[:, i] = (self.y[:, obs.species] * obs.coefficients).sum(1)

        # calculate expressions
        obs_names = self.model.observables.keys()
        obs_dict = dict((k, self.yobs[k]) for k in obs_names)
        for expr in self.model.expressions_dynamic():
            expr_subs = expr.expand_expr().subs(subs)
            func = sympy.lambdify(obs_names, expr_subs, "numpy")
            self.yexpr[expr.name] = func(**obs_dict)

def odesolve(model, tspan, param_values=None, y0=None, integrator='vode', cleanup=True, verbose=False,
             **integrator_options):
    """Integrate a model's ODEs over a given timespan.

    This is a simple function-based interface to integrating (a.k.a. solving or
    simulating) a model. If you need to integrate a model repeatedly with
    different parameter values or initial conditions (as in parameter
    estimation), using the Solver class directly will provide much better
    performance.

    Parameters
    ----------
    model : pysb.Model
        Model to integrate.
    tspan : vector-like
        Time values over which to integrate. The first and last values define
        the time range, and the returned trajectories will be sampled at every
        value.
    param_values : vector-like, optional
        Values to use for every parameter in the model. Ordering is determined
        by the order of model.parameters. If not specified, parameter values
        will be taken directly from model.parameters.
    y0 : vector-like, optional
        Values to use for the initial condition of all species. Ordering is
        determined by the order of model.species. If not specified, initial
        conditions will be taken from model.initial_conditions (with initial
        condition parameter values taken from `param_values` if specified).
    integrator : string, optional
        Name of the integrator to use, taken from the list of integrators known
        to :py:class:`scipy.integrate.ode`.
    integrator_params
        Additional parameters for the integrator.

    Returns
    -------
    yfull : record array
        The trajectories calculated by the integration. The first dimension is
        time and its length is identical to that of `tspan`. The second
        dimension is species/observables and its length is the sum of the
        lengths of model.species and model.observables. The dtype of the array
        specifies field names: '__s0', '__s1', etc. for the species and
        observable names for the observables. See Notes below for further
        explanation and caveats.

    Notes
    -----
    This function was the first implementation of integration support and
    accordingly it has a few warts:

    * It performs expensive code generation every time it is called.

    * The returned array, with its record-style data-type, allows convenient
      selection of individual columns by their field names, but does not permit
      slice ranges or indexing by integers for columns. If you only need access
      to your model's observables this is usually not a problem, but sometimes
      it's more convenient to have a "regular" array. See Examples below for
      code to do this.

    The actual integration code has since been moved to the Solver class and
    split up such that the code generation is only performed on
    initialization. The model may then be integrated repeatedly with different
    parameter values or initial conditions with much better
    performance. Additionally, Solver makes the species trajectories available
    as a simple array and only uses the record array for the observables where
    it makes sense.

    This function now simply serves as a wrapper for creating a Solver object,
    calling its ``run`` method, and building the record array to return.

    Examples
    --------
    Simulate a model and display the results for an observable:

    >>> from pysb.examples.robertson import model
    >>> from numpy import linspace
    >>> numpy.set_printoptions(precision=4)
    >>> yfull = odesolve(model, linspace(0, 40, 10))
    >>> print(yfull['A_total'])            #doctest: +NORMALIZE_WHITESPACE
    [ 1.      0.899   0.8506  0.8179  0.793   0.7728  0.7557  0.7408  0.7277
    0.7158]

    Obtain a view on a returned record array which uses an atomic data-type and
    integer indexing (note that the view's data buffer is shared with the
    original array so there is no extra memory cost):

    >>> print(yfull.shape)
    (10,)
    >>> print(yfull.dtype)                 #doctest: +NORMALIZE_WHITESPACE
    [('__s0', '<f8'), ('__s1', '<f8'), ('__s2', '<f8'), ('A_total', '<f8'),
    ('B_total', '<f8'), ('C_total', '<f8')]
    >>> print(yfull[0:4, 1:3])             #doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    IndexError: too many indices...
    >>> yarray = yfull.view(float).reshape(len(yfull), -1)
    >>> print(yarray.shape)
    (10, 6)
    >>> print(yarray.dtype)
    float64
    >>> print(yarray[0:4, 1:3])
    [[  0.0000e+00   0.0000e+00]
     [  2.1672e-05   1.0093e-01]
     [  1.6980e-05   1.4943e-01]
     [  1.4502e-05   1.8209e-01]]

    """

    solver = Solver(model, tspan, cleanup=cleanup,
                    verbose=verbose, **integrator_options)
    solver.run(param_values, y0)

    species_names = ['__s%d' % i for i in range(solver.y.shape[1])]
    yfull_dtype = list(zip(species_names, itertools.repeat(float)))
    if len(solver.yobs.dtype):
        yfull_dtype += solver.yobs.dtype.descr
    if len(solver.yexpr.dtype):
        yfull_dtype += solver.yexpr.dtype.descr
    yfull = numpy.ndarray(len(solver.y), yfull_dtype)

    n_sp = solver.y.shape[1]
    n_ob = solver.yobs_view.shape[1]
    n_ex = solver.yexpr_view.shape[1]

    yfull_view = yfull.view(float).reshape(len(yfull), -1)
    yfull_view[:,:n_sp] = solver.y
    yfull_view[:,n_sp:n_sp+n_ob] = solver.yobs_view
    yfull_view[:,n_sp+n_ob:n_sp+n_ob+n_ex] = solver.yexpr_view

    return yfull


def setup_module(module):
    """Doctest fixture for nose."""
    # Distutils' temp directory creation code has a more-or-less unsuppressable
    # print to stdout which will break the doctest which triggers it (via
    # scipy.weave.inline). So here we run an end-to-end test of the inlining
    # system to get that print out of the way at a point where it won't matter.
    # As a bonus, the test harness is suppressing writes to stdout at this time
    # anyway so the message is just swallowed silently.
    Solver._test_inline()
