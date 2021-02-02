from pysb.simulator.base import Simulator, SimulationResult
import scipy.integrate, scipy.sparse
from sympy.utilities.autowrap import CythonCodeWrapper
from sympy.utilities.codegen import (
    C99CodeGen, Routine, InputArgument, OutputArgument, default_datatypes
)
import distutils
import pysb.bng
import sympy
import re
from functools import partial
import numpy as np
import warnings
import os
import inspect
from pysb.logging import get_logger, PySBModelLoggerAdapter, EXTENDED_DEBUG
import logging
import contextlib
import importlib
import tempfile
import shutil
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
        * ``cleanup``: Boolean, whether to delete temporary files.

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
        cleanup = kwargs.pop('cleanup', True)
        with_jacobian = kwargs.pop('use_analytic_jacobian', False)
        integrator = kwargs.pop('integrator', 'vode')
        compiler = kwargs.pop('compiler', None)
        integrator_options = kwargs.pop('integrator_options', {})

        if kwargs:
            raise ValueError('Unknown keyword argument(s): {}'.format(
                ', '.join(kwargs.keys())
            ))
        # Generate the equations for the model
        pysb.bng.generate_equations(self._model, cleanup, self.verbose)

        builder_cls = _select_rhs_builder(compiler, self._logger)
        self._logger.debug("Using RhsBuilder: %s", builder_cls.__name__)
        self.rhs_builder = builder_cls(
            model, with_jacobian, cleanup=cleanup, _logger=self._logger
        )

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

        if num_processors == 1:
            self._logger.debug('Single processor (serial) mode')
        else:
            self._logger.debug('Multi-processor (parallel) mode using {} '
                               'processes'.format(num_processors))

        with SerialExecutor() if num_processors == 1 else \
                ProcessPoolExecutor(max_workers=num_processors) as executor:
            sim_partial = partial(
                _integrator_process,
                tspan=self.tspan,
                integrator_name=self._init_kwargs.get('integrator', 'vode'),
                integrator_opts=self.opts,
                rhs_builder=self.rhs_builder,
            )
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


def _integrator_process(
    initials, param_values, tspan, integrator_name, integrator_opts, rhs_builder
):
    """ Single integrator process, for parallel execution """
    rhs_fn = rhs_builder.rhs_fn
    jac_fn = rhs_builder.jacobian_fn
    initials = np.array(initials, float)
    p = np.array(param_values, float)
    e = rhs_builder.calc_expressions_constant(param_values)
    extra_args = (p, e)

    # LSODA
    if integrator_name == 'lsoda':
        return scipy.integrate.odeint(
            rhs_fn,
            initials,
            tspan,
            args=extra_args,
            Dfun=jac_fn,
            tfirst=True,
            **integrator_opts
        )

    # All other integrators
    integrator = scipy.integrate.ode(rhs_fn, jac=jac_fn)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', 'No integrator name match')
        integrator.set_integrator(integrator_name, **integrator_opts)
    integrator.set_initial_value(initials, tspan[0])

    # Set parameter vectors for RHS func and Jacobian
    integrator.set_f_params(*extra_args)
    if rhs_builder.with_jacobian:
        integrator.set_jac_params(*extra_args)

    trajectory = np.ndarray((len(tspan), rhs_builder.num_species))
    trajectory[0] = initials
    i = 1
    while integrator.successful() and integrator.t < tspan[-1]:
        trajectory[i] = integrator.integrate(tspan[i])
        i += 1
    if integrator.t < tspan[-1]:
        trajectory[i:, :] = 'nan'

    return trajectory


class RhsBuilder:
    """Provides the ODE right-hand side evaluation function, given a model.

    Also provides the Jacobian of the RHS function (partial derivatives of
    species concentrations with respect to all other species).

    This is an abstract base class; concrete subclasses must implement the
    `_get_rhs` and `_get_jacobian` methods. These methods shall return function
    objects for evaluating the RHS and Jacobian, respectively, of the system of
    ODEs. Subclasses may also implement the `check` classmethod which should
    return a bool indicating whether it is usable in the current environment
    (it could for example check that all dependencies are present and working).

    All implementations of this class must be picklable so they can be sent
    across to the ProcessPoolExecutor workers in separate processes. The
    `_get_rhs` and `_get_jacobian` methods may return process-specific
    non-picklable values.

    Parameters
    ----------
    model : pysb.Model
        Model to simulate.
    with_jacobian : bool, optional (default: False)
        Whether to construct the Jacobian evaluation function.
    cleanup : bool, optional (default: True)
        Whether to delete the work directory with generated code and compiled
        output files upon destruction.

    Attributes
    ----------
    with_jacobian : bool
        Whether to construct the Jacobian evaluation function.
    cleanup : bool
        Whether to delete the work directory with generated code and compiled
        output files upon destruction.
    y : sympy.MatrixSymbol
        Symbol "y" representing the amount of all model species.
    p : sympy.MatrixSymbol
        Symbol "p" representing all model parameters.
    e : sympy.MatrixSymbol
        Symbol "e" representing all model constant expressions.
    o : sympy.MatrixSymbol
        Symbol "o" representing all model observables.
    kinetics : sympy.Matrix
        Symbolic form of model reaction rates in terms of y, p, e, o, and time.
    kinetics_jacobian_y : sympy.SparseMatrix
        Symbolic form of partial derivatives of kinetics with respect to y
        (Only set if with_jacobian is True)
    kinetics_jacobian_o : sympy.SparseMatrix
        Symbolic form of partial derivatives of kinetics with respect to o
        (Only set if with_jacobian is True)
    stoichiometry_matrix : scipy.sparse.csr_matrix
        The model's stoichiometry matrix.
    observables_matrix : scipy.sparse.csr_matrix
        Encodes the linear combinations of species that define the model's
        observables.
    model_name : str
        The name of the model.

    """

    def __init__(self, model, with_jacobian=False, cleanup=True, _logger=None):
        self.with_jacobian = with_jacobian
        self.cleanup = cleanup
        self._logger = _logger
        expr_dynamic = model.expressions_dynamic(include_derived=True)
        expr_constant = model.expressions_constant(include_derived=True)
        self.y = sympy.MatrixSymbol('y', len(model.species), 1)
        self.p = sympy.MatrixSymbol('p', len(model.parameters_all()), 1)
        self.e = sympy.MatrixSymbol('e', len(expr_constant), 1)
        self.o = sympy.MatrixSymbol('o', len(model.observables), 1)
        # Parameters symbols. We also need this independently of all_subs.
        param_subs = dict(zip(model.parameters_all(), self.p))
        # All substitution rules we need to apply to the reaction expressions.
        all_subs = param_subs.copy()
        # Species symbols.
        all_subs.update({
            sympy.Symbol('__s%d' % i): self.y[i]
            for i in range(len(model.species))
        })
        # Observables symbols.
        all_subs.update(dict(zip(model.observables, self.o)))
        # Constant expressions symbols.
        all_subs.update(dict(zip(expr_constant, self.e)))
        # Dynamic expressions (expanded).
        all_subs.update({e: e.expand_expr() for e in expr_dynamic})
        self.kinetics = sympy.Matrix([
            r['rate'].subs(all_subs) for r in model.reactions
        ])
        if with_jacobian:
            # The Jacobian can be quite large but it's extremely sparse. We can
            # obtain a sparse representation by making a SparseMatrix copy of
            # the kinetics vector and computing the Jacobian on that.
            kinetics_sparse = sympy.SparseMatrix(self.kinetics)
            # Rather than substitute all the observables linear-combination-of-
            # species expressions into the kinetics and then compute the
            # Jacobian, we can use the total derivative / chain rule to
            # simplify things. The final jacobian of the kinetics must be
            # computed as jac(y) + jac(o) * observables_matrix.
            self._logger.debug("Computing Jacobian matrices")
            self.kinetics_jacobian_y = kinetics_sparse.jacobian(self.y)
            self.kinetics_jacobian_o = kinetics_sparse.jacobian(self.o)
        obs_matrix = scipy.sparse.lil_matrix(
            (len(model.observables), len(model.species)), dtype=np.int64
        )
        for i, obs in enumerate(model.observables):
            obs_matrix[i, obs.species] = obs.coefficients
        self.observables_matrix = obs_matrix.tocsr()
        self.stoichiometry_matrix = model.stoichiometry_matrix
        self.model_name = model.name
        self._expressions_constant = [
            e.expand_expr().xreplace(param_subs) for e in expr_constant
        ]
        self._work_path = None
        self._expressions_constant_fn = None
        self._rhs_fn = None
        self._jacobian_fn = None

    def __del__(self):
        if self._work_path:
            if self.cleanup:
                shutil.rmtree(self._work_path, ignore_errors=True)
                self._logger.debug("Removed work dir: %s", self._work_path)
            else:
                self._logger.debug(
                    "Leaving work dir in place (cleanup=False): %s",
                    self._work_path
                )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Clear cached function objects (will be regenerated on demand).
        state["_expressions_constant_fn"] = None
        state["_rhs_fn"] = None
        state["_jacobian_fn"] = None
        # Loggers are not picklable in Python 3.6.
        state["_logger_extra"] = state["_logger"].extra
        del state["_logger"]
        return state

    def __setstate__(self, state):
        state["_logger"] = PySBModelLoggerAdapter(
            get_logger(self.__module__), state["_logger_extra"]
        )
        del state["_logger_extra"]
        self.__dict__.update(state)

    @property
    def work_path(self):
        """Location for saving any temporary working files."""
        if self._work_path is None:
            # Only initialize this on demand since it will create a directory
            # which requires cleanup on destruction.
            self._work_path = tempfile.mkdtemp(suffix="_pysb_compile")
            self._logger.debug("Created work dir: %s", self._work_path)
        return self._work_path

    @property
    def num_species(self):
        """Number of species in the model."""
        return self.stoichiometry_matrix.shape[0]

    @property
    def num_reactions(self):
        """Number of reactions in the model."""
        return self.stoichiometry_matrix.shape[1]

    @property
    def num_observables(self):
        """Number of observables in the model."""
        return self.observables_matrix.shape[0]

    @property
    def rhs_fn(self):
        """The RHS function rhs(time, y, params, const_exprs)."""
        if self._rhs_fn is None:
            self._logger.debug("Constructing rhs function")
            self._rhs_fn = self._get_rhs()
        return self._rhs_fn

    @property
    def jacobian_fn(self):
        """The Jacobian function jac(time, y, params, const_exprs).

        The value is None if with_jacobian is False."""
        if self.with_jacobian and self._jacobian_fn is None:
            self._logger.debug("Constructing jacobian function")
            self._jacobian_fn = self._get_jacobian()
        return self._jacobian_fn

    def calc_expressions_constant(self, p):
        """Compute constant expressions vector e from parameters vector p."""
        if self._expressions_constant_fn is None:
            # This function is expected to be called rarely enough (once per
            # simulation) that we'll just create it here with lambdify and not
            # allow implementations to override it.
            self._logger.debug("Constructing constant expressions function")
            self._expressions_constant_fn = sympy.lambdify(
                [self.p], self._expressions_constant
            )
        e = self._expressions_constant_fn(p[:, None])
        return np.array(e, float)[:, None]

    def _get_rhs(self):
        """Return the RHS function rhs(time, y, params, const_exprs).

        All subclasses must implement this method."""
        raise NotImplementedError

    def _get_jacobian(self):
        """Return the Jacobian function jac(time, y, params, const_exprs).

        Subclasses may implement this method for Jacobian support."""
        raise NotImplementedError

    @classmethod
    def check(cls):
        """Raises an exception if this class is not usable.

        The default implementation does nothing (always succeeds)."""
        return True

    @classmethod
    def check_safe(cls):
        """Returns a boolean indicating whether this class is usable."""
        try:
            cls.check()
            return True
        except Exception:
            return False


class PythonRhsBuilder(RhsBuilder):

    def _get_rhs(self):
        kinetics = sympy.lambdify(
            [self.y, self.p, self.e, self.o], self.kinetics
        )

        def rhs(t, y, p, e):
            o = (self.observables_matrix * y)[:, None]
            v = kinetics(y[:, None], p[:, None], e, o)
            ydot = self.stoichiometry_matrix * v[:, 0]
            return ydot

        return rhs

    def _get_jacobian(self):
        kinetics_jacobian_y = sympy.lambdify(
            [self.y, self.p, self.e, self.o], self.kinetics_jacobian_y
        )
        kinetics_jacobian_o = sympy.lambdify(
            [self.y, self.p, self.e, self.o], self.kinetics_jacobian_o
        )

        def jacobian(t, y, p, e):
            o = (self.observables_matrix * y)[:, None]
            dy = kinetics_jacobian_y(y[:, None], p[:, None], e, o)
            do = kinetics_jacobian_o(y[:, None], p[:, None], e, o)
            # Compute Jacobian of the kinetics vector by total derivative.
            jv = dy + do * self.observables_matrix
            jac = (self.stoichiometry_matrix * jv).todense()
            return jac

        return jacobian


def _mat_sym_dims(symbol):
    """Return codegen Argument dimensions for a MatrixSymbol."""
    return ((0, symbol.shape[0] - 1), (0, symbol.shape[1] - 1))


class CythonRhsBuilder(RhsBuilder):

    def __init__(self, model, with_jacobian=False, cleanup=True, _logger=None):
        super(CythonRhsBuilder, self).__init__(
            model, with_jacobian, cleanup, _logger
        )
        routine_names = ["kinetics"]
        if with_jacobian:
            routine_names += ["kinetics_jacobian_y", "kinetics_jacobian_o"]
        # We want more control over various details of code generation and
        # wrapper module creation than sympy's autowrap provides, so we'll use
        # the lower-level building blocks directly.
        routines = {name: self._build_routine(name) for name in routine_names}
        code_gen = C99CodeGen()
        extra_compile_args = [
            # The RHS evaluation is such a tiny part of overall integration
            # time, even for huge models, that compiler optimization actually
            # takes more time than it will ever yield back. Since Cython sets
            # -O2 by default we need to override it.
            "-O0",
        ]
        # Opt in to the newer numpy C API which is only supported in Cython 3+.
        import Cython
        if not Cython.__version__.startswith("0."):
            extra_compile_args.append(
                "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
            )
        code_wrapper = CythonCodeWrapper(
            code_gen,
            filepath=self.work_path,
            extra_compile_args=extra_compile_args,
        )
        # Build a module-name-safe string that identifies the model as uniquely
        # as possible to assist in debugging.
        escaped_name = re.sub(
            r"[^A-Za-z0-9]",
            "_",
            self.model_name.encode("unicode_escape").decode(),
        )
        base_name = "pysb_" + escaped_name + "_kinetics"
        code_wrapper._filename = base_name
        code_wrapper._module_basename = base_name + "_wrapper"
        self._logger.debug("Running code generation and Cython compilation")
        functions = {
            name: code_wrapper.wrap_code(routine)
            for name, routine in routines.items()
        }
        # Grab specs for the Cython-compiled modules for reloading later.
        self.module_specs = {
            name: inspect.getmodule(function).__spec__
            for name, function in functions.items()
        }

    def _build_routine(self, name):
        expr = getattr(self, name)
        out = sympy.MatrixSymbol("out", *expr.shape)
        routine = Routine(
            name,
            [
                InputArgument(self.y, dimensions=_mat_sym_dims(self.y)),
                InputArgument(self.p, dimensions=_mat_sym_dims(self.p)),
                InputArgument(self.e, dimensions=_mat_sym_dims(self.e)),
                InputArgument(self.o, dimensions=_mat_sym_dims(self.o)),
                OutputArgument(
                    out,
                    out,
                    expr,
                    datatype=default_datatypes["float"],
                    dimensions=_mat_sym_dims(out),
                ),
            ],
            [],
            [],
            [],
        )
        return routine

    def _load_function(self, name):
        # Import one of our previously compiled Cython functions.
        spec = self.module_specs[name]
        # These two lines seem strangely redundant, but it's how specs work...
        module = spec.loader.create_module(spec)
        spec.loader.exec_module(module)
        function = getattr(module, name + "_c")
        return function

    # The inner functions below are, admittedly, basically identical to those
    # in PythonRhsBuilder. Initial implementations differed more but they
    # converged over time. They have not been refactored out into a common
    # implementation to allow for exploration of performance improvements.

    def _get_rhs(self):
        kinetics = self._load_function("kinetics")

        def rhs(t, y, p, e):
            o = (self.observables_matrix * y)[:, None]
            v = kinetics(y[:, None], p[:, None], e, o)
            ydot = self.stoichiometry_matrix * v[:, 0]
            return ydot

        return rhs

    def _get_jacobian(self):
        kinetics_jacobian_y = self._load_function("kinetics_jacobian_y")
        kinetics_jacobian_o = self._load_function("kinetics_jacobian_o")

        def jacobian(t, y, p, e):
            o = (self.observables_matrix * y)[:, None]
            dy = kinetics_jacobian_y(y[:, None], p[:, None], e, o)
            do = kinetics_jacobian_o(y[:, None], p[:, None], e, o)
            # Compute Jacobian of the kinetics vector by total derivative.
            jv = dy + do * self.observables_matrix
            jac = self.stoichiometry_matrix * jv
            return jac

        return jacobian

    _check_ok = False

    @classmethod
    def check(cls):
        # Quick return if previous check succeeded, otherwise re-run the checks
        # so the exception is always raised if called repeatedly.
        if cls._check_ok:
            return
        compiler_exc = None
        try:
            import Cython
            Cython.inline("x = 1", force=True, quiet=True)
            cls._check_ok = True
            return
        except ImportError:
            raise RuntimeError(
                "Please install the Cython package to use this compiler."
            )
        # Catch some common C compiler configuration problems so we can raise a
        # chained exception with a more helpful message.
        except (
            Cython.Compiler.Errors.CompileError,
            distutils.errors.CCompilerError,
            distutils.errors.DistutilsPlatformError,
        ) as e:
            compiler_exc = e
        except ValueError as e:
            if e.args == ("Symbol table not found",):
                # This is a common error building against numpy with mingw32.
                compiler_exc = e
            else:
                raise
        message = "Please check you have a functional C compiler"
        if os.name == "nt":
            message += ", available from " \
                       "https://wiki.python.org/moin/WindowsCompilers"
        else:
            message += "."
        # Reference chained exceptions for specifics on what went wrong.
        message += "\nSee the above error messages for more details."
        raise RuntimeError(message) from compiler_exc


# Keep these sorted in priority order.
_rhs_builders = {
    "cython": CythonRhsBuilder,
    "python": PythonRhsBuilder,
}


def _select_rhs_builder(compiler, logger):
    if compiler is None:
        # Probe for the first (best) working builder.
        for cls in _rhs_builders.values():
            if cls.check_safe():
                break
        else:
            raise RuntimeError("No usable ODE compiler found.")
        if cls is PythonRhsBuilder:
            logger.warning(
                "This system of ODEs will be evaluated in pure Python. "
                "This may be slow for large models. We recommend "
                "installing the 'cython' package for compiling the ODEs to "
                "C code. This warning can be suppressed by specifying "
                "compiler='python'."
            )
    else:
        try:
            cls = _rhs_builders[compiler]
        except KeyError:
            raise ValueError('Unknown ODE compiler name: %s' % compiler) \
                from None
        cls.check()
    return cls


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
