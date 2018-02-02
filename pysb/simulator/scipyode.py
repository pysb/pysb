from pysb.simulator.base import Simulator, SimulationResult
import scipy.integrate
try:
    # weave is not available under Python 3.
    from weave import inline as weave_inline
    import weave.build_tools
    import distutils.errors
except ImportError:
    weave_inline = None
try:
    import theano.tensor
    from sympy.printing.theanocode import theano_function
except ImportError:
    theano = None
try:
    import Cython
except ImportError:
    Cython = None
from sympy.printing.lambdarepr import lambdarepr
import distutils
import pysb.bng
import sympy
import re
import numpy as np
import warnings
import os
from pysb.logging import get_logger, EXTENDED_DEBUG
import logging
import itertools
import contextlib
import importlib


CYTHON_DECL = '#cython: boundscheck=False, wraparound=False, ' \
              'nonecheck=False, initializedcheck=False\n'


class ScipyOdeSimulator(Simulator):
    """
    Simulate a model using SciPy ODE integration

    Uses :func:`scipy.integrate.odeint` for the ``lsoda`` integrator,
    :func:`scipy.integrate.ode` for all other integrators.

    .. warning::
        The interface for this class is considered experimental and may
        change without warning as PySB is updated.

    Parameters
    ----------
    model : pysb.Model
        Model to simulate.
    tspan : vector-like, optional
        Time values over which to simulate. The first and last values define
        the time range. Returned trajectories are sampled at every value unless
        the simulation is interrupted for some reason, e.g., due to
        satisfaction of a logical stopping criterion (see 'tout' below).
    initials : vector-like or dict, optional
        Values to use for the initial condition of all species. Ordering is
        determined by the order of model.species. If not specified, initial
        conditions will be taken from model.initial_conditions (with
        initial condition parameter values taken from `param_values` if
        specified).
    param_values : vector-like or dict, optional
        Values to use for every parameter in the model. Ordering is
        determined by the order of model.parameters.
        If passed as a dictionary, keys must be parameter names.
        If not specified, parameter values will be taken directly from
        model.parameters.
    verbose : bool or int, optional (default: False)
        Sets the verbosity level of the logger. See the logging levels and
        constants from Python's logging module for interpretation of integer
        values. False is equal to the PySB default level (currently WARNING),
        True is equal to DEBUG.
    **kwargs : dict
        Extra keyword arguments, including:

        * ``integrator``: Choice of integrator, including ``vode`` (default),
          ``zvode``, ``lsoda``, ``dopri5`` and ``dop853``. See
          :func:`scipy.integrate.ode` for further information.
        * ``integrator_options``: A dictionary of keyword arguments to
          supply to the integrator. See :func:`scipy.integrate.ode`.
        * ``compiler``: Choice of compiler for ODE system: ``cython``,
          ``weave`` (Python 2 only), ``theano`` or ``python``. Leave
          unspecified or equal to None for auto-select (tries weave,
          then cython, then python). Cython, weave and theano all compile the
          equation system into C code. Python is the slowest but most
          compatible.
        * ``cleanup``: Boolean, `cleanup` argument used for
          :func:`pysb.bng.generate_equations` call

    Notes
    -----
    If ``tspan`` is not defined, it may be defined in the call to the
    ``run`` method.

    Examples
    --------
    Simulate a model and display the results for an observable:

    >>> from pysb.examples.robertson import model
    >>> import numpy as np
    >>> np.set_printoptions(precision=4)
    >>> sim = ScipyOdeSimulator(model, tspan=np.linspace(0, 40, 10))
    >>> simulation_result = sim.run()
    >>> print(simulation_result.observables['A_total']) \
        #doctest: +NORMALIZE_WHITESPACE
    [1.      0.899   0.8506  0.8179  0.793   0.7728  0.7557  0.7408  0.7277
    0.7158]

    For further information on retrieving trajectories (species,
    observables, expressions over time) from the ``simulation_result``
    object returned by :func:`run`, see the examples under the
    :class:`SimulationResult` class.
    """

    _supports = {'multi_initials': True,
                 'multi_param_values': True}

    # some sane default options for a few well-known integrators
    default_integrator_options = {
        'vode': {
            'method': 'bdf',
            'with_jacobian': True,
            # Set nsteps as high as possible to give our users flexibility in
            # choosing their time step. (Let's be safe and assume vode was
            # compiled with 32-bit ints. What would actually happen if it was
            # and we passed 2**64-1 though?)
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
        compiler_mode = kwargs.get('compiler', None)
        # Generate the equations for the model
        pysb.bng.generate_equations(self._model, self.cleanup, self.verbose)

        # ODE RHS -----------------------------------------------
        self._eqn_subs = {e: e.expand_expr(expand_observables=True) for
                          e in self._model.expressions}
        ode_mat = sympy.Matrix(self.model.odes).subs(self._eqn_subs)

        if compiler_mode is None:
            self._compiler = self._autoselect_compiler()
            if self._compiler == 'python':
                self._logger.warning(
                    "This system of ODEs will be evaluated in pure Python. "
                    "This may be slow for large models. We recommend "
                    "installing a package for compiling the ODEs to C code: "
                    "'weave' (recommended for Python 2) or "
                    "'cython' (recommended for Python 3). This warning can "
                    "be suppressed by specifying compiler_mode='python'.")
            self._logger.debug('Equation mode set to "%s"' % self._compiler)
        else:
            self._compiler = compiler_mode

        extra_compile_args = []
        # Inhibit weave C compiler warnings unless log level <= EXTENDED_DEBUG.
        # Note that since the output goes straight to stderr rather than via the
        # logging system, the threshold must be lower than DEBUG or else the
        # Nose logcapture plugin will cause the warnings to be shown and tests
        # will fail due to unexpected output.
        if not self._logger.isEnabledFor(EXTENDED_DEBUG):
            extra_compile_args.append('-w')

        # Use lambdarepr (Python code) with Cython, otherwise use C code
        eqn_repr = lambdarepr if self._compiler == 'cython' else sympy.ccode

        if self._compiler in ('weave', 'cython'):
            # Prepare the string representations of the RHS equations

            code_eqs = '\n'.join(['ydot[%d] = %s;' %
                                  (i, eqn_repr(o))
                                  for i, o in enumerate(ode_mat)])
            code_eqs = str(self._eqn_substitutions(code_eqs))

            # Allocate ydot here, once.
            ydot = np.zeros(len(self.model.species))

            if self._compiler == 'cython':
                if not Cython:
                    raise ImportError('Cython library is not installed')
                code_eqs = CYTHON_DECL + code_eqs

                def rhs(t, y, p):
                    # note that the evaluated code sets ydot as a side effect
                    Cython.inline(code_eqs, quiet=True)

                    return ydot

                with _set_cflags_no_warnings(self._logger):
                    rhs(0.0, self.initials[0], self.param_values[0])
            else:
                # Weave
                if not weave_inline:
                    raise ImportError('Weave library is not installed')
                for arr_name in ('ydot', 'y', 'p'):
                    macro = arr_name.upper() + '1'
                    code_eqs = re.sub(r'\b%s\[(\d+)\]' % arr_name,
                                      '%s(\\1)' % macro, code_eqs)

                def rhs(t, y, p):
                    # note that the evaluated code sets ydot as a side effect
                    weave_inline(code_eqs, ['ydot', 't', 'y', 'p'],
                                 extra_compile_args=extra_compile_args)
                    return ydot

                # Call rhs once just to trigger the weave C compilation step
                # while asserting control over distutils logging.
                with self._patch_distutils_logging:
                    rhs(0.0, self.initials[0], self.param_values[0])

        elif self._compiler in ('theano', 'python'):
            self._symbols = sympy.symbols(','.join('__s%d' % sp_id for sp_id in
                                                   range(len(
                                                       self.model.species)))
                                          + ',') + tuple(model.parameters)

            if self._compiler == 'theano':
                if theano is None:
                    raise ImportError('Theano library is not installed')

                code_eqs_py = theano_function(
                    self._symbols,
                    [o if not o.is_zero else theano.tensor.zeros(1)
                     for o in ode_mat],
                    on_unused_input='ignore'
                )
            else:
                code_eqs_py = sympy.lambdify(self._symbols,
                                             sympy.flatten(ode_mat))

            def rhs(t, y, p):
                return code_eqs_py(*itertools.chain(y, p))
        else:
            raise ValueError('Unknown compiler_mode: %s' % self._compiler)

        # JACOBIAN -----------------------------------------------
        # We'll keep the code for putting together the matrix in Sympy
        # in case we want to do manipulations of the matrix later (e.g., to
        # put together the sensitivity matrix)
        jac_fn = None
        if self._use_analytic_jacobian:
            species_symbols = [sympy.Symbol('__s%d' % i)
                               for i in range(len(self._model.species))]
            jac_matrix = ode_mat.jacobian(species_symbols)

            if self._compiler == 'theano':
                jac_eqs_py = theano_function(
                    self._symbols,
                    [j if not j.is_zero else theano.tensor.zeros(1)
                     for j in jac_matrix],
                    on_unused_input='ignore'
                )

                def jacobian(t, y, p):
                    jacmat = np.asarray(jac_eqs_py(*itertools.chain(y, p)))
                    jacmat.shape = (len(self.model.odes),
                                    len(self.model.species))
                    return jacmat

            elif self._compiler in ('weave', 'cython'):
                # Prepare the stringified Jacobian equations.
                jac_eqs_list = []
                for i in range(jac_matrix.shape[0]):
                    for j in range(jac_matrix.shape[1]):
                        entry = jac_matrix[i, j]
                        # Skip zero entries in the Jacobian
                        if entry == 0:
                            continue
                        jac_eq_str = 'jac[%d, %d] = %s;' % (
                            i, j, eqn_repr(entry))
                        jac_eqs_list.append(jac_eq_str)
                jac_eqs = str(self._eqn_substitutions('\n'.join(jac_eqs_list)))

                # Allocate jac array here, once, and initialize to zeros.
                jac = np.zeros(
                    (len(self._model.odes), len(self._model.species)))

                if self._compiler == 'weave':
                    # Substitute array refs with calls to the JAC1 macro
                    jac_eqs = re.sub(r'\bjac\[(\d+), (\d+)\]',
                                     r'JAC2(\1, \2)', jac_eqs)
                    # Substitute calls to the Y1 and P1 macros
                    for arr_name in ('y', 'p'):
                        macro = arr_name.upper() + '1'
                        jac_eqs = re.sub(r'\b%s\[(\d+)\]' % arr_name,
                                         '%s(\\1)' % macro, jac_eqs)

                    def jacobian(t, y, p):
                        weave_inline(jac_eqs, ['jac', 't', 'y', 'p'],
                                     extra_compile_args=extra_compile_args)
                        return jac

                    # Manage distutils logging, as above for rhs.
                    with self._patch_distutils_logging:
                        jacobian(0.0, self.initials[0], self.param_values[0])
                else:
                    jac_eqs = CYTHON_DECL + jac_eqs

                    def jacobian(t, y, p):
                        Cython.inline(jac_eqs, quiet=True)
                        return jac

                    with _set_cflags_no_warnings(self._logger):
                        jacobian(0.0, self.initials[0], self.param_values[0])
            else:
                jac_eqs_py = sympy.lambdify(self._symbols, jac_matrix, "numpy")

                def jacobian(t, y, p):
                    return jac_eqs_py(*itertools.chain(y, p))

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

    @property
    def _patch_distutils_logging(self):
        """Return distutils logging context manager based on our logger."""
        return _patch_distutils_logging(self._logger.logger)

    @classmethod
    def _test_inline(cls):
        """
        Detect whether weave.inline is functional.

        Produces compile warnings, which we suppress by capturing STDERR.
        """
        if not hasattr(cls, '_use_inline'):
            cls._use_inline = False
            if weave_inline is not None:
                logger = get_logger(__name__)
                extra_compile_args = []
                # See comment in __init__ for why this must be EXTENDED_DEBUG.
                if not logger.isEnabledFor(EXTENDED_DEBUG):
                    if os.name == 'posix':
                        extra_compile_args.append('2>/dev/null')
                    elif os.name == 'nt':
                        extra_compile_args.append('2>NUL')
                try:
                    with _patch_distutils_logging(logger):
                        weave_inline('int i=0; i=i;', force=1,
                                     extra_compile_args=extra_compile_args)
                    cls._use_inline = True
                except (weave.build_tools.CompileError,
                        distutils.errors.CompileError, ImportError):
                    pass

    @classmethod
    def _test_cython(cls):
        if not hasattr(cls, '_use_cython'):
            cls._use_cython = False
            if Cython is None:
                return
            try:
                Cython.inline('x = 1', force=True, quiet=True)
                cls._use_cython = True
            except Cython.Compiler.Errors.CompileError:
                pass

    @classmethod
    def _autoselect_compiler(cls):
        """ Auto-select equation backend """

        # Try weave
        cls._test_inline()
        if cls._use_inline:
            return 'weave'

        # Try cython
        cls._test_cython()
        if cls._use_cython:
            return 'cython'

        # Default to python/lambdify
        return 'python'

    def _eqn_substitutions(self, eqns):
        """String substitutions on the sympy C code for the ODE RHS and
        Jacobian functions to use appropriate terms for variables and
        parameters."""
        # Substitute 'y[i]' for 'si'
        eqns = re.sub(r'\b__s(\d+)\b',
                      lambda m: 'y[%s]' % (int(m.group(1))),
                      eqns)

        # Substitute 'p[i]' for any named parameters
        for i, p in enumerate(self._model.parameters):
            eqns = re.sub(r'\b(%s)\b' % p.name, 'p[%d]' % i, eqns)
        return eqns

    def run(self, tspan=None, initials=None, param_values=None):
        """
        Run a simulation and returns the result (trajectories)

        .. note::
            In early versions of the Simulator class, ``tspan``, ``initials``
            and ``param_values`` supplied to this method persisted to future
            :func:`run` calls. This is no longer the case.

        Parameters
        ----------
        tspan
        initials
        param_values
            See parameter definitions in :class:`ScipyOdeSimulator`.

        Returns
        -------
        A :class:`SimulationResult` object
        """
        super(ScipyOdeSimulator, self).run(tspan=tspan,
                                           initials=initials,
                                           param_values=param_values,
                                           _run_kwargs=[])
        n_sims = len(self.param_values)
        trajectories = np.ndarray((n_sims, len(self.tspan),
                              len(self._model.species)))
        for n in range(n_sims):
            self._logger.info('Running simulation %d of %d', n + 1, n_sims)
            if self.integrator == 'lsoda':
                trajectories[n] = scipy.integrate.odeint(
                    self.func,
                    self.initials[n],
                    self.tspan,
                    Dfun=self.jac_fn,
                    args=(self.param_values[n],),
                    **self.opts)
            else:
                self.integrator.set_initial_value(self.initials[n],
                                                  self.tspan[0])
                # Set parameter vectors for RHS func and Jacobian
                self.integrator.set_f_params(self.param_values[n])
                if self._use_analytic_jacobian:
                    self.integrator.set_jac_params(self.param_values[n])
                trajectories[n][0] = self.initials[n]
                i = 1
                while self.integrator.successful() and self.integrator.t < \
                        self.tspan[-1]:
                    self._logger.log(EXTENDED_DEBUG,
                                     'Simulation %d/%d Integrating t=%g',
                                     n + 1, n_sims, self.integrator.t)
                    trajectories[n][i] = self.integrator.integrate(self.tspan[i])
                    i += 1
                if self.integrator.t < self.tspan[-1]:
                    trajectories[n, i:, :] = 'nan'

        tout = np.array([self.tspan]*n_sims)
        self._logger.info('All simulation(s) complete')
        return SimulationResult(self, tout, trajectories)


@contextlib.contextmanager
def _patch_distutils_logging(base_logger):
    """Patch distutils logging functionality with logging.Logger calls.

    The value of the 'base_logger' argument should be a logging.Logger instance,
    and its effective level will be passed on to the patched distutils loggers.

    distutils.log contains its own internal PEP 282 style logging system that
    sends messages straight to stdout/stderr, and numpy.distutils.log extends
    that. This code patches all of this with calls to logging.LoggerAdapter
    instances, and disables the module-level threshold-setting functions so we
    can retain full control over the threshold. Also all WARNING messages are
    "downgraded" to INFO to suppress excessive use of WARNING-level logging in
    numpy.distutils.

    """
    logger = get_logger(__name__)
    logger.debug('patching distutils and numpy.distutils logging')
    logger_methods = 'log', 'debug', 'info', 'warn', 'error', 'fatal'
    other_functions = 'set_threshold', 'set_verbosity'
    saved_symbols = {}
    for module_name in 'distutils.log', 'numpy.distutils.log':
        new_logger = _DistutilsProxyLoggerAdapter(
            base_logger, {'module': module_name}
        )
        module = importlib.import_module(module_name)
        # Save the old values.
        for name in logger_methods + other_functions:
            saved_symbols[module, name] = getattr(module, name)
        # Replace logging functions with bound methods of the Logger object.
        for name in logger_methods:
            setattr(module, name, getattr(new_logger, name))
        # Replace threshold-setting functions with no-ops.
        for name in other_functions:
            setattr(module, name, lambda *args, **kwargs: None)
    try:
        yield
    finally:
        logger.debug('restoring distutils and numpy.distutils logging')
        # Restore everything we overwrote.
        for (module, name), value in saved_symbols.items():
            setattr(module, name, value)


@contextlib.contextmanager
def _set_cflags_no_warnings(logger):
    """ Suppress cython warnings by setting -w flag """
    del_cflags = False
    if 'CFLAGS' not in os.environ \
            and not logger.isEnabledFor(EXTENDED_DEBUG):
        del_cflags = True
        os.environ['CFLAGS'] = '-w'
    try:
        yield
    finally:
        if del_cflags:
            del os.environ['CFLAGS']


class _DistutilsProxyLoggerAdapter(logging.LoggerAdapter):
    """A logging adapter for the distutils logging patcher."""
    def process(self, msg, kwargs):
        return '(from %s) %s' % (self.extra['module'], msg), kwargs
    # Map 'warn' to 'info' to reduce chattiness.
    warn = logging.LoggerAdapter.info
    # Provide 'fatal' to match up with distutils log functions.
    fatal = logging.LoggerAdapter.critical
