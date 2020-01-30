from pysb.simulator.base import Simulator, SimulationResult
import scipy.integrate
try:
    import Cython
except ImportError:
    Cython = None
from sympy.printing.lambdarepr import lambdarepr
import distutils
import pysb.bng
import sympy
import re
from functools import partial
import numpy as np
import warnings
import os
from pysb.logging import get_logger, EXTENDED_DEBUG
import logging
import itertools
import contextlib
import importlib
from concurrent.futures import ProcessPoolExecutor, Executor, Future


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
        conditions will be taken from model.initials (with initial condition
        parameter values taken from `param_values` if specified).
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
          or ``python``. Leave unspecified or equal to None for auto-select
          (tries cython, then python). Cython compiles the
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

    default_cython_directives = {
        'boundscheck': False,
        'wraparound': False,
        'nonecheck': False,
        'initializedcheck': False
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
        self._use_analytic_jacobian = kwargs.pop('use_analytic_jacobian',
                                                 False)
        self.cleanup = kwargs.pop('cleanup', True)
        integrator = kwargs.pop('integrator', 'vode')
        compiler_mode = kwargs.pop('compiler', None)
        integrator_options = kwargs.pop('integrator_options', {})

        if kwargs:
            raise ValueError('Unknown keyword argument(s): {}'.format(
                ', '.join(kwargs.keys())
            ))
        # Generate the equations for the model
        pysb.bng.generate_equations(self._model, self.cleanup, self.verbose)

        # ODE RHS -----------------------------------------------
        self._eqn_subs = {e: e.expand_expr(expand_observables=True) for
                          e in self._model.expressions}
        self._eqn_subs.update({e: e.expand_expr(expand_observables=True) for
                               e in self._model._derived_expressions})
        ode_mat = sympy.Matrix(self.model.odes).subs(self._eqn_subs)

        if compiler_mode is None:
            self._compiler = self._autoselect_compiler()
            if self._compiler == 'python':
                self._logger.warning(
                    "This system of ODEs will be evaluated in pure Python. "
                    "This may be slow for large models. We recommend "
                    "installing the 'cython' package for compiling the ODEs to "
                    "C code. This warning can be suppressed by specifying "
                    "compiler='python'.")
            self._logger.debug('Equation mode set to "%s"' % self._compiler)
        else:
            self._compiler = compiler_mode

        self._compiler_directives = None

        # Use lambdarepr (Python code) with Cython, otherwise use C code
        eqn_repr = lambdarepr if self._compiler == 'cython' else sympy.ccode

        if self._compiler == 'cython':
            # Prepare the string representations of the RHS equations

            code_eqs = '\n'.join(['ydot[%d] = %s;' %
                                  (i, eqn_repr(o))
                                  for i, o in enumerate(ode_mat)])
            code_eqs = str(self._eqn_substitutions(code_eqs))

            # Allocate ydot here, once.
            ydot = np.zeros(len(self.model.species))

            self._compiler_directives = kwargs.pop(
                'cython_directives', self.default_cython_directives
            )

            if not Cython:
                raise ImportError('Cython library is not installed')

            rhs = _get_rhs(self._compiler,
                           code_eqs,
                           ydot=ydot,
                           compiler_directives=self._compiler_directives
                           )

            with _set_cflags_no_warnings(self._logger):
                rhs(0.0, self.initials[0], self.param_values[0])

            self._code_eqs = code_eqs

        elif self._compiler == 'python':
            self._symbols = sympy.symbols(','.join('__s%d' % sp_id for sp_id in
                                                   range(len(
                                                       self.model.species)))
                                          + ',') + tuple(model.parameters)

            self._code_eqs = (self._symbols, sympy.flatten(ode_mat))
        else:
            raise ValueError('Unknown compiler_mode: %s' % self._compiler)

        # JACOBIAN -----------------------------------------------
        # We'll keep the code for putting together the matrix in Sympy
        # in case we want to do manipulations of the matrix later (e.g., to
        # put together the sensitivity matrix)
        self._jac_eqs = None
        if self._use_analytic_jacobian:
            species_symbols = [sympy.Symbol('__s%d' % i)
                               for i in range(len(self._model.species))]
            jac_matrix = ode_mat.jacobian(species_symbols)

            if self._compiler == 'cython':
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
                if '# Not supported in Python' in jac_eqs:
                    raise ValueError('Analytic Jacobian calculation failed')

                # Allocate jac array here, once, and initialize to zeros.
                jac = np.zeros(
                    (len(self._model.odes), len(self._model.species)))

                jac_fn = _get_rhs(
                    self._compiler,
                    jac_eqs,
                    compiler_directives=self._compiler_directives,
                    jac=jac
                )

                with _set_cflags_no_warnings(self._logger):
                    jac_fn(0.0, self.initials[0], self.param_values[0])
                self._jac_eqs = jac_eqs
            else:
                self._jac_eqs = (self._symbols, jac_matrix, "numpy")

        # build integrator options list from our defaults and any kwargs
        # passed to this function
        options = {}
        if self.default_integrator_options.get(integrator):
            options.update(
                self.default_integrator_options[integrator])  # default options

        options.update(integrator_options)  # overwrite
        # defaults
        self.opts = options

        if integrator != 'lsoda':
            # Only used to check the user has selected a valid integrator
            self.integrator = scipy.integrate.ode(None)
            with warnings.catch_warnings():
                warnings.filterwarnings('error', 'No integrator name match')
                self.integrator.set_integrator(integrator, **options)

    @property
    def _patch_distutils_logging(self):
        """Return distutils logging context manager based on our logger."""
        return _patch_distutils_logging(self._logger.logger)

    @classmethod
    def _test_cython(cls):
        if not hasattr(cls, '_use_cython'):
            cls._use_cython = False
            if Cython is None:
                return
            try:
                Cython.inline('x = 1', force=True, quiet=True)
                cls._use_cython = True
            except (Cython.Compiler.Errors.CompileError,
                    distutils.errors.DistutilsPlatformError,
                    ValueError) as e:
                if not cls._check_compiler_error(e, 'cython'):
                    raise

    @staticmethod
    def _check_compiler_error(e, compiler):
        if isinstance(e, distutils.errors.DistutilsPlatformError) and \
                str(e) != 'Unable to find vcvarsall.bat':
            return False

        if isinstance(e, ValueError) and e.args != ('Symbol table not found',):
            return False

        # Build platform-specific C compiler error message
        message = 'Please check you have a functional C compiler'
        if os.name == 'nt':
            message += ', available from ' \
                       'https://wiki.python.org/moin/WindowsCompilers'
        else:
            message += '.'

        get_logger(__name__).warn(
            '{} compiler appears non-functional. {}\n'
            'Original error: {}'.format(compiler, message, repr(e)))

        return True

    @classmethod
    def _autoselect_compiler(cls):
        """ Auto-select equation backend """

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
        for i, p in enumerate(self._model._derived_parameters):
            eqns = re.sub(r'\b(%s)\b' % p.name,
                          'p[%d]' % (i + len(self._model.parameters)), eqns)
        return eqns

    def run(self, tspan=None, initials=None, param_values=None,
            num_processors=1):
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
        num_processors : int
            Number of processes to use (default: 1). Set to a larger number
            (e.g. the number of CPU cores available) for parallel execution of
            simulations. This is only useful when simulating with more than one
            set of initial conditions and/or parameters.

        Returns
        -------
        A :class:`SimulationResult` object
        """
        super(ScipyOdeSimulator, self).run(tspan=tspan,
                                           initials=initials,
                                           param_values=param_values,
                                           _run_kwargs=[])
        n_sims = len(self.param_values)

        num_species = len(self._model.species)
        num_odes = len(self._model.odes)
        if num_processors == 1:
            self._logger.debug('Single processor (serial) mode')
        else:
            self._logger.debug('Multi-processor (parallel) mode using {} '
                               'processes'.format(num_processors))

        with SerialExecutor() if num_processors == 1 else \
                ProcessPoolExecutor(max_workers=num_processors) as executor:
            sim_partial = partial(_integrator_process, code_eqs=self._code_eqs, jac_eqs=self._jac_eqs,
                                  num_species=num_species, num_odes=num_odes, tspan=self.tspan,
                                  integrator_name=self._init_kwargs.get('integrator', 'vode'),
                                  compiler=self._compiler, integrator_opts=self.opts,
                                  compiler_directives=self._compiler_directives)

            results = [executor.submit(sim_partial, *args)
                       for args in zip(self.initials, self.param_values)]
            try:
                trajectories = [r.result() for r in results]
            finally:
                for r in results:
                    r.cancel()

        tout = np.array([self.tspan] * n_sims)
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


def _get_rhs(compiler, code_eqs, ydot=None, jac=None, compiler_directives=None):
    if compiler == 'cython':
        if 'math.' in code_eqs:
            code_eqs = 'import math\n' + code_eqs

        def rhs(t, y, p):
            # note that the evaluated code sets ydot as a side effect
            Cython.inline(code_eqs, quiet=True,
                          cython_compiler_directives=compiler_directives)

            return ydot if ydot is not None else jac
    else:
        def rhs(t, y, p):
            return code_eqs(*itertools.chain(y, p))

    return rhs


def _integrator_process(initials, param_values, code_eqs, jac_eqs, num_species, num_odes, tspan,
                        integrator_name, compiler, integrator_opts, compiler_directives):
    """ Single integrator process, for parallel execution """
    if compiler == 'python':
        code_eqs = sympy.lambdify(*code_eqs)
        if jac_eqs:
            jac_eqs = sympy.lambdify(*jac_eqs)

    rhs = _get_rhs(compiler, code_eqs, ydot=np.zeros(num_species),
                   compiler_directives=compiler_directives)

    jac_fn = None
    if jac_eqs:
        jac_eqs = _get_rhs(compiler,
                           jac_eqs,
                           jac=np.zeros((num_odes, num_species)),
                           compiler_directives=compiler_directives)

    # LSODA
    if integrator_name == 'lsoda':
        return scipy.integrate.odeint(
            rhs,
            initials,
            tspan,
            args=(param_values, ),
            Dfun=jac_fn,
            tfirst=True,
            **integrator_opts
        )

    # All other integrators
    integrator = scipy.integrate.ode(rhs, jac=jac_fn)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', 'No integrator name match')
        integrator.set_integrator(integrator_name, **integrator_opts)
    integrator.set_initial_value(initials, tspan[0])

    # Set parameter vectors for RHS func and Jacobian
    integrator.set_f_params(param_values)
    if jac_eqs:
        integrator.set_jac_params(param_values)

    trajectory = np.ndarray((len(tspan), num_species))
    trajectory[0] = initials
    i = 1
    while integrator.successful() and integrator.t < tspan[-1]:
        trajectory[i] = integrator.integrate(tspan[i])
        i += 1
    if integrator.t < tspan[-1]:
        trajectory[i:, :] = 'nan'

    return trajectory


class SerialExecutor(Executor):
    """ Execute tasks in serial (immediately on submission) """
    def submit(self, fn, *args, **kwargs):
        f = Future()
        try:
            result = fn(*args, **kwargs)
        except BaseException as e:
            f.set_exception(e)
        else:
            f.set_result(result)

        return f
