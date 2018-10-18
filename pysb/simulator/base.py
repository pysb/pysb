from abc import ABCMeta, abstractmethod
import numpy as np
import itertools
import sympy
import collections
import numbers
from pysb.core import MonomerPattern, ComplexPattern, as_complex_pattern, \
                      Parameter, Expression
from pysb.logging import get_logger, EXTENDED_DEBUG
import pickle
from pysb import __version__ as PYSB_VERSION
from datetime import datetime
import dateutil.parser
import copy
from warnings import warn
from pysb.pattern import SpeciesPatternMatcher
try:
    basestring
except NameError:
    # Python 3 compatibility.
    basestring = str

try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import h5py
except ImportError:
    h5py = None


class SimulatorException(Exception):
    pass


class Simulator(object):
    """An abstract base class for numerical simulation of models.

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
        satisfaction
        of a logical stopping criterion (see 'tout' below).
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

    Attributes
    ----------
    verbose: bool
        Verbosity flag passed to the constructor.
    model : pysb.Model
        Model passed to the constructor.
    tspan : vector-like
        Time values passed to the constructor.

    Notes
    -----
    If ``tspan`` is not defined, it may be defined in the call to the
    ``run`` method.

    The dimensionality of ``tout`` depends on whether a single simulation
    or multiple simulations are run.

    The dimensionalities of ``y``, ``yobs``, ``yobs_view``, ``yexpr``, and
    ``yexpr_view`` depend on the number of simulations being run as well
    as on the type of simulation, i.e., spatial vs. non-spatial.

    """
    __metaclass__ = ABCMeta

    _supports = { 'multi_initials' : False,
                  'multi_param_values' : False }

    @abstractmethod
    def __init__(self, model, tspan=None, initials=None,
                 param_values=None, verbose=False, **kwargs):
        # Get or create base a PySB logger for this module and model
        self._logger = get_logger(self.__module__, model=model,
                                  log_level=verbose)
        self._logger.debug('Simulator created')
        self._model = model
        self.verbose = verbose
        self.tout = None
        # Per-run initial conditions/parameter/tspan override
        self._tspan = tspan
        # Base initials and param values
        self._initials = None
        self.initials = initials
        self._params = None
        self.param_values = param_values
        # Per-run tspan, initials and param_values
        self._run_tspan = None
        self._run_initials = None
        self._run_params = None
        # Store init kwargs and run kwargs if needed for saving results
        self._init_kwargs = kwargs
        self._run_kwargs = None

    @property
    def model(self):
        return self._model

    @property
    def tspan(self):
        return self._run_tspan if self._run_tspan is not None else self._tspan

    @tspan.setter
    def tspan(self, new_tspan):
        self._tspan = new_tspan

    @staticmethod
    def _num_sims_calc(initials_or_params):
        """ Calculate number of simulations implied by initials or param
        values """
        if initials_or_params is None:
            return None

        if isinstance(initials_or_params, np.ndarray):
            return len(initials_or_params)

        first_entry = next(iter(initials_or_params.values()))

        try:
            return len(first_entry)  # First entry is iterable
        except TypeError:
            return 1  # First entry is non-iterable, e.g. int, float

    @property
    def initials_length(self):
        try:
            return len(self.initials)
        except SimulatorException:
            # Network free simulators
            if self._initials:
                return len(list(self._initials.values())[0])
            else:
                return 1

    def _update_initials_dict(self, initials_dict, initials_source):
        if isinstance(initials_source, collections.Mapping):
            # Can't just use .update() as we need to test
            # equality with .is_equivalent_to()
            for cp, val in initials_source.items():
                found = False
                for existing_cp in initials_dict.keys():
                    if existing_cp.is_equivalent_to(as_complex_pattern(cp)):
                        initials_dict[existing_cp] = val
                        found = True
                        break
                if not found:
                    initials_dict[cp] = val
        elif initials_source is not None:
            # Update from array-like structure, which we can only do if we
            # have the species available (e.g. not in network-free simulations)
            if not self.model.species:
                raise ValueError(
                    'Cannot update initials from an array-like source without '
                    'model species. ')
            initials_dict = {}
            for cp_idx, cp in enumerate(self.model.species):
                initials_dict[cp] = [initials_source[n][cp_idx] for n in
                                     range(len(initials_source))]
        return initials_dict

    @property
    def initials_dict(self):
        initials_dict = {cp: [param.value] for cp, param in
                         self.model.initial_conditions}
        # Apply any base initial overrides
        initials_dict = self._update_initials_dict(initials_dict,
                                                   self._initials)
        # Apply any per-run initial overrides
        initials_dict = self._update_initials_dict(initials_dict,
                                                   self._run_initials)

        return initials_dict

    @property
    def initials(self):
        if not self.model.species:
            raise SimulatorException('No model species list - either '
                                     'generate the model equations or use '
                                     'initials_dict() for network-free '
                                     'simulations')

        # Check potential quick return options
        if self._run_initials is not None:
            if not isinstance(self._run_initials, collections.Mapping) and \
                    self._initials is None:
                return self._run_initials
        elif not isinstance(self._initials, collections.Mapping) and \
                self._initials is not None:
            return self._initials

        # Otherwise, build the list from the model, and any overrides
        # specified in self._initials and self._run_initials
        n_sims_initials = self._num_sims_calc(self._initials)
        n_sims_run = self._num_sims_calc(self._run_initials)

        if n_sims_initials is not None and n_sims_run is not None \
                and n_sims_run != n_sims_initials:
            raise ValueError(
                "The base initials set with self.initials imply {} "
                "simulations, but the run() initials imply {} simulations."
                " Either set self.initials=None, or change the number of "
                "simulations in the run() initials".format(
                    n_sims_initials, n_sims_run))
        if n_sims_initials is None:
            if n_sims_run is None:
                n_sims_initials = 1
            else:
                n_sims_initials = n_sims_run

        # At this point (after dimensionality check), we can return
        # self._run_initials if it's not a dictionary and not None
        if self._run_initials is not None and not isinstance(
                self._run_initials, collections.Mapping):
            return self._run_initials

        n_sims_params = len(self.param_values)
        n_sims_actual = max(n_sims_params, n_sims_initials)

        y0 = np.full((n_sims_actual, len(self.model.species)), np.nan)

        # Process any overrides
        y0 = self._update_y0(y0, self._run_initials)
        y0 = self._update_y0(y0, self._initials)

        # Fast NaN check with short circuit
        if np.isnan(np.sum(y0)):
            # Get remaining initials from the model itself and
            # self.param_values, if necessary
            subs = None
            if self._model.expressions:
                # Only need parameter substitutions if model has expressions
                subs = [
                    dict((p, pv[i]) for i, p in
                         enumerate(self._model.parameters))
                    for pv in self.param_values]
                if len(subs) == 1 and n_sims_actual > 1:
                    subs = list(itertools.repeat(subs[0], n_sims_actual))
            y0 = self._update_y0(y0, self._model.initial_conditions, subs,
                                 n_sims_params)

        # Any remaining unset initials should be set to zero
        y0 = np.nan_to_num(y0)

        return y0

    def _update_y0(self, y0, initials_source, subs=None, n_sims_params=None):
        """ Update the initial conditions list y0 using initials_source """
        if initials_source is None:
            return y0

        if isinstance(initials_source, np.ndarray) and \
                initials_source.shape != 1:
            # If initials_source is a multi-dimensional array, we can set
            # the y0 values directly
            nan_pos = np.isnan(y0)
            y0[nan_pos] = initials_source[nan_pos]
            return y0

        if isinstance(initials_source, collections.Mapping):
            initials_source = initials_source.items()

        for cp, value_obj in initials_source:
            cp = as_complex_pattern(cp)
            si = self._model.get_species_index(cp)
            if si is None:
                raise IndexError("Species not found in model: %s" % repr(cp))

            # Loop over all simulations
            for sim in range(len(y0)):
                # If this initial condition has already been set, skip it
                if not np.isnan(y0[sim][si]):
                    continue

                if isinstance(value_obj, (collections.Sequence, np.ndarray))\
                        and isinstance(value_obj[sim], numbers.Number):
                    value = value_obj[sim]
                elif isinstance(value_obj, Expression):
                    value = value_obj.expand_expr().evalf(subs=subs[sim])
                elif isinstance(value_obj, Parameter):
                    if sim > 0 and n_sims_params == 1:
                        # Parameters can be copied from previous
                        # simulation if they have not been specified
                        # explicitly in self.param_values
                        value = y0[sim - 1][si]
                    else:
                        # Set parameter using param_values
                        pi = self._model.parameters.index(value_obj)
                        value = self.param_values[sim][pi]
                else:
                    raise TypeError("Unexpected initial condition "
                                    "value type: %s" % type(value_obj))

                y0[sim][si] = value

        return y0

    @initials.setter
    def initials(self, new_initials):
        self._initials = self._process_incoming_initials(new_initials)

    def _process_incoming_initials(self, new_initials):
        if new_initials is None:
            return None

        # Check if new_initials is a dict, and if so validate the keys
        # (ComplexPatterns)
        if isinstance(new_initials, dict):
            n_sims = 1
            if len(new_initials) > 0:
                n_sims = self._num_sims_calc(new_initials)
            for cplx_pat, val in new_initials.items():
                if not isinstance(cplx_pat, (MonomerPattern,
                                             ComplexPattern)):
                    raise ValueError('Dictionary key %s is not a '
                                     'MonomerPattern or ComplexPattern' %
                                     repr(cplx_pat))
                # if val is a number, convert it to a single-element array
                if not isinstance(val, (collections.Sequence, np.ndarray)):
                    val = [val]
                    new_initials[cplx_pat] = np.array(val)
                # otherwise, check whether simulator supports multiple
                # initial values :
                if len(val) != n_sims:
                    raise ValueError("all arrays in new_initials dictionary "
                                     "must be equal length")
                if not np.isfinite(val).all():
                    raise ValueError('Please check initial {} for non-finite '
                                     'values'.format(cplx_pat))
        else:
            if not isinstance(new_initials, np.ndarray):
                new_initials = np.array(new_initials, copy=False)
            # if new_initials is a 1D array, convert to a 2D array of length 1
            if len(new_initials.shape) == 1:
                new_initials = np.resize(new_initials, (1, len(new_initials)))
            n_sims = new_initials.shape[0]
            # make sure number of initials values equals len(model.species)
            if new_initials.shape[1] != len(self._model.species):
                raise ValueError("new_initials must be the same length as "
                                 "model.species")
            if not np.isfinite(new_initials).all():
                raise ValueError('Please check initials array '
                                 'for non-finite values')

        if n_sims > 1:
            if not self._supports['multi_initials']:
                raise ValueError(
                    self.__class__.__name__ +
                    " does not support multiple initial values at this time.")
            if 1 < len(self.param_values) != n_sims:
                raise ValueError(
                    'Cannot set initials for {} simulations '
                    'when param_values has been set for {} '
                    'simulations'.format(
                        n_sims, len(self.param_values)))

        return new_initials

    @property
    def param_values(self):
        if self._params is not None and \
                not isinstance(self._params, dict) and \
                self._run_params is None:
            return self._params
        elif self._run_params is not None and \
                not isinstance(self._run_params, dict) and \
                self._params is None:
            return self._run_params

        # create parameter vector from the values in the model
        param_values_dict = {}
        n_sims = self._num_sims_calc(self._params)
        if isinstance(self._params, dict):
            param_values_dict.update(self._params)
        elif isinstance(self._params, np.ndarray):
            param_values_dict = dict(zip(
                [p.name for p in self._model.parameters], self._params.T))

        n_sims_run = self._num_sims_calc(self._run_params)

        if n_sims is None:
            n_sims = n_sims_run
        elif n_sims_run is not None and n_sims_run != n_sims:
            raise ValueError(
                "The base parameters set with self.param_values imply "
                "{} simulations, but the run() params imply {} "
                "simulations. Either set self.param_values=None, or "
                "change the number of simulations in the run() params"
                .format(n_sims, n_sims_run))

        # At this point (after dimensionality check) we can return the
        # _run_params, if it's not a dict
        if self._run_params is not None:
            if not isinstance(self._run_params, dict):
                return self._run_params
            else:
                param_values_dict.update(self._run_params)

        if n_sims is None:
            n_sims = 1

        # Get the base parameters from the model
        param_values = np.array([p.value for p in self._model.parameters])
        param_values = np.repeat([param_values], n_sims, axis=0)
        # Process overrides
        for key in param_values_dict.keys():
            try:
                pi = self._model.parameters.index(
                                self._model.parameters[key])
            except KeyError:
                raise IndexError("new_params dictionary has unknown "
                                 "parameter name (%s)" % key)
            # loop over n_sims
            for n in range(n_sims):
                param_values[n][pi] = param_values_dict[key][n]

        # return array
        return param_values

    @param_values.setter
    def param_values(self, new_params):
        self._params = self._process_incoming_params(new_params)

    def _process_incoming_params(self, new_params):
        if new_params is None:
            return None
        if isinstance(new_params, dict):
            n_sims = 1
            if len(new_params) > 0:
                n_sims = self._num_sims_calc(new_params)
            for key, val in new_params.items():
                if key not in self._model.parameters.keys():
                    raise IndexError("new_params dictionary has unknown "
                                     "parameter name (%s)" % key)
                # if val is a number, convert it to a single-element array
                if not isinstance(val, collections.Sequence):
                    val = [val]
                    new_params[key] = np.array(val)
                # Check all elements are the same length
                if len(val) != n_sims:
                    raise ValueError("all arrays in params dictionary "
                                     "must be equal length")
        else:
            if not isinstance(new_params, np.ndarray):
                new_params = np.array(new_params)
            # if new_params is a 1D array, convert to a 2D array of length 1
            if len(new_params.shape) == 1:
                new_params = np.resize(new_params, (1, len(new_params)))
            n_sims = new_params.shape[0]
            # make sure number of param values equals len(model.parameters)
            if new_params.shape[1] != len(self._model.parameters):
                raise ValueError("new_params must be the same length as "
                                 "model.parameters")

        # Check whether simulator supports multiple param_values
        if n_sims > 1 and not self._supports['multi_param_values']:
            raise ValueError(
                self.__class__.__name__ +
                " does not support multiple parameter values at this time.")
        return new_params

    def _reset_run_overrides(self):
        """
        Reset any single-run tspan, initials, param_values

        When calling run(), the user can specify tspan, initials and
        param_values, which are only used for a single run. This method
        resets those overrides after the run is complete (called from
        :func:`SimulationResult.__init__`).
        """
        self._run_tspan = None
        self._run_initials = None
        self._run_params = None

    @abstractmethod
    def run(self, tspan=None, initials=None, param_values=None,
            _run_kwargs=None):
        """Run a simulation.

        Notes for developers implementing Simulator subclasses:

        Implementations should return a :class:`.SimulationResult` object.
        Subclasses should pass any additional arguments run as a dictonary
        to the `_run_kwargs` argument when calling the superclass's run
        method. If the run method has variable keyword arguments, this can
        be achieved by passing `_run_kwargs=locals()` to the superclass's
        run method. The run kwargs are used for reference when saving and
        loading SimulationResults to disk. They aren't compulsory, but not
        including them will generate a warning. To suppress (e.g. if there
        are no additional arguments), set `_run_kwargs=[]`.
        """
        self._logger.info('Simulation(s) started')
        if _run_kwargs:
            # Don't store these arguments twice
            _run_kwargs.pop('self')
            _run_kwargs.pop('initials', None)
            _run_kwargs.pop('param_values', None)
            _run_kwargs.pop('tspan', None)
            self._run_kwargs = _run_kwargs
        elif _run_kwargs is None:
            self._logger.warning(
                '{} has not passed any additional run arguments to '
                '_run_kwargs. Instructions are included in the Simulation '
                'base class run method docstring.'.format(
                    self.__class__.__name__))
        self._run_tspan = tspan
        if self.tspan is None:
            raise ValueError("tspan must be defined before "
                             "simulation can run")
        self._run_params = self._process_incoming_params(param_values)
        self._run_initials = self._process_incoming_initials(initials)

        # If only one set of param_values, run all simulations
        # with the same parameters
        if len(self.param_values) == 1 and self.initials_length > 1:
            new_params = np.repeat(self.param_values,
                                   self.initials_length,
                                   axis=0)
            if self._run_params is None:
                self._params = new_params
            else:
                self._run_params = new_params

        # Error checks on 'param_values' and 'initials'
        if len(self.param_values) != self.initials_length:
            raise ValueError(
                    "'param_values' and 'initials' must be equal lengths.\n"
                    "len(param_values): %d\n"
                    "len(initials): %d" %
                    (len(self.param_values), self.initials_length))
        elif len(self.param_values.shape) != 2 or \
                self.param_values.shape[1] != len(self._model.parameters):
            raise ValueError(
                    "'param_values' must be a 2D array of dimension N_SIMS x "
                    "len(model.parameters).\n"
                    "param_values.shape: " + str(self.param_values.shape) +
                    "\nlen(model.parameters): %d" %
                    len(self._model.parameters))

        if self.model.species and (len(self.initials.shape) != 2 or
                self.initials.shape[1] != len(self._model.species)):
            raise ValueError(
                    "'initials' must be a 2D array of dimension N_SIMS x "
                    "len(model.species).\n"
                    "initials.shape: " + str(self.initials.shape) +
                    "\nlen(model.species): %d" % len(self._model.species))

        return None


class SimulationResult(object):
    """
    Results of a simulation with properties and methods to access them.

    .. warning::
        Please note that the interface for this class is considered
        experimental and may change without warning as PySB is updated.

    Notes
    -----
    In the attribute descriptions, a "trajectory set" is a 2D numpy array,
    species on first axis and time on second axis, with each element
    containing the concentration or count of the species at the specified time.

    A list of trajectory sets contains a trajectory set for each simulation.

    Parameters
    ----------
    simulator : Simulator
        The simulator object that generated the trajectories
    tout: list-like
        Time points returned by the simulator (may be different from ``tspan``
        if simulation is interrupted for some reason).
    trajectories : list or numpy.ndarray
        A set of species trajectories from a simulation. Should either be a
        list of 2D numpy arrays or a single 3D numpy array.
    squeeze : bool, optional (default: True)
        Return trajectories as a 2D array, rather than a 3d array, if only
        a single simulation was performed.
    simulations_per_param_set : int
        Number of trajectories per parameter set. Typically always 1 for
        deterministic simulators (e.g. ODE), but with stochastic simulators
        multiple trajectories per parameter/initial condition set are often
        desired.
    model: pysb.Model
    initials: numpy.ndarray
    param_values: numpy.ndarray
        model, initials, param_values are an alternative constructor
        mechanism used when loading SimulationResults from files (see
        :func:`SimulationResult.load`). Setting just the simulator argument
        instead of these arguments is recommended.

    Examples
    --------
    The following examples use a simple model with three observables and one
    expression, with a single simulation.

    >>> from pysb.examples.expression_observables import model
    >>> from pysb.simulator import ScipyOdeSimulator
    >>> import numpy as np
    >>> np.set_printoptions(precision=4)
    >>> sim = ScipyOdeSimulator(model, tspan=np.linspace(0, 40, 10), \
                                integrator_options={'atol': 1e-20})
    >>> simulation_result = sim.run()

    ``simulation_result`` is a :class:`SimulationResult` object. An
    observable can be accessed like so:

    >>> print(simulation_result.observables['Bax_c0']) \
        #doctest: +NORMALIZE_WHITESPACE
    [1.0000e+00   1.1744e-02   1.3791e-04   1.6196e-06   1.9020e-08
     2.2337e-10   2.6232e-12   3.0806e-14   3.6178e-16   4.2492e-18]

    It is also possible to retrieve the value of all observables at a
    particular time point, e.g. the final concentrations:

    >>> print(simulation_result.observables[-1]) \
        #doctest: +SKIP
    (4.2492e-18,   1.6996e-16,  1.)

    Expressions are read in the same way as observables:

    >>> print(simulation_result.expressions['NBD_signal']) \
        #doctest: +NORMALIZE_WHITESPACE
    [0.   4.7847  4.9956  4.9999  5.   5.   5.   5.   5.   5. ]

    The species trajectories can be accessed as a numpy ndarray:

    >>> print(simulation_result.species) #doctest: +NORMALIZE_WHITESPACE
    [[1.0000e+00   0.0000e+00   0.0000e+00]
     [1.1744e-02   5.2194e-02   9.3606e-01]
     [1.3791e-04   1.2259e-03   9.9864e-01]
     [1.6196e-06   2.1595e-05   9.9998e-01]
     [1.9020e-08   3.3814e-07   1.0000e+00]
     [2.2337e-10   4.9637e-09   1.0000e+00]
     [2.6232e-12   6.9951e-11   1.0000e+00]
     [3.0806e-14   9.5840e-13   1.0000e+00]
     [3.6178e-16   1.2863e-14   1.0000e+00]
     [4.2492e-18   1.6996e-16   1.0000e+00]]

    Species, observables and expressions can be combined into a single numpy
    ndarray and accessed similarly. Here, the initial concentrations of all
    these entities are examined:

    >>> print(simulation_result.all[0]) #doctest: +SKIP
    ( 1.,  0.,  0.,  1.,  0.,  0.,  0.)

    The ``all`` array can be accessed as a pandas DataFrame object,
    which allows for more convenient indexing and access to pandas advanced
    functionality, such as indexing and slicing. Here, the concentrations of
    the observable ``Bax_c0`` and the expression ``NBD_signal`` are read at
    time points between 5 and 15 seconds:

    >>> df = simulation_result.dataframe
    >>> print(df.loc[5:15, ['Bax_c0', 'NBD_signal']]) \
        #doctest: +NORMALIZE_WHITESPACE
                 Bax_c0  NBD_signal
    time
    8.888889   0.000138    4.995633
    13.333333  0.000002    4.999927
    """
    CUSTOM_ATTR_PREFIX = 'usrattr_'

    def __init__(self, simulator, tout, trajectories=None,
                 observables_and_expressions=None, squeeze=True,
                 simulations_per_param_set=1,
                 model=None, initials=None, param_values=None):
        if simulator:
            simulator._logger.debug('SimulationResult constructor started')
            self._param_values = simulator.param_values.copy()
            try:
                self._initials = simulator.initials.copy()
            except SimulatorException:
                # Network free simulations don't have initials list, only dict
                self._initials = simulator.initials_dict.copy()
            self._model = copy.deepcopy(simulator._model)
            self.simulator_class = simulator.__class__
            self.init_kwargs = copy.deepcopy(simulator._init_kwargs)
            self.run_kwargs = copy.deepcopy(simulator._run_kwargs)
        else:
            self._param_values = param_values
            self._initials = initials
            self._model = model
            self.simulator_class = None
            self.init_kwargs = {}
            self.run_kwargs = {}

        self.squeeze = squeeze
        self.tout = np.asarray(tout)
        self._yfull = None
        self.n_sims_per_parameter_set = simulations_per_param_set
        self.pysb_version = PYSB_VERSION
        self.timestamp = datetime.now()
        self.custom_attrs = {}

        if trajectories is None and observables_and_expressions is None:
            raise ValueError('Need to supply at least one of species '
                             'trajectories or observables_and_expressions')

        if trajectories is not None and len(trajectories) > 0:
            # Validate incoming trajectories
            if getattr(trajectories, 'ndim', None) == 3:
                # trajectories is a 3D array, create a list of 2D arrays
                # This is just a view and doesn't copy the data
                self._y = [tr for tr in trajectories]
            else:
                # Not a 3D array, check for a list of 2D arrays
                try:
                    if any(tr.ndim != 2 for tr in trajectories):
                        raise AttributeError
                except (AttributeError, TypeError):
                    raise ValueError("trajectories should be a 3D array or a "
                                     "list of 2D arrays")
                self._y = trajectories

            self._nsims = len(self._y)
            if len(self.tout) != self.nsims:
                raise ValueError("Simulator tout should be the same length as "
                                 "trajectories")
            for i in range(self.nsims):
                if len(self.tout[i]) != self._y[i].shape[0]:
                    raise ValueError("The number of time points in tout[{0}] "
                                     "should match the trajectories array for "
                                     "simulation {0}".format(i))
                if self._y[i].shape[1] != len(self._model.species):
                    raise ValueError("The number of species in trajectory {0} "
                                     "should match length of "
                                     "model.species".format(i))
        else:
            self._y = None

        # Calculate ``yobs`` and ``yexpr`` based on values of ``y``
        exprs = self._model.expressions_dynamic()
        expr_names = [expr.name for expr in exprs]
        model_obs = self._model.observables
        obs_names = list(model_obs.keys())
        param_names = list(p.name for p in self._model.parameters)

        if not _allow_unicode_recarray():
            for name_list, name_type in zip(
                    (expr_names, obs_names, param_names),
                    ('Expression', 'Observable', 'Parameter')):
                for i, name in enumerate(name_list):
                    try:
                        name_list[i] = name.encode('ascii')
                    except UnicodeEncodeError:
                        error_msg = 'Non-ASCII compatible ' + \
                                    '%s names not allowed' % name_type
                        raise ValueError(error_msg)

        yobs_dtype = (list(zip(obs_names, itertools.repeat(float)))
                      if obs_names else float)
        yexpr_dtype = (list(zip(expr_names, itertools.repeat(float)))
                       if expr_names else float)

        if observables_and_expressions:
            # Observables and expression values are used as supplied
            self._nsims = len(observables_and_expressions)
            self._yobs_view = [observables_and_expressions[n][:, 0:(len(
                self._model.observables))] for n in range(self.nsims)]
            self._yexpr_view = [observables_and_expressions[n][:, (len(
                self._model.observables)):] for n in range(self.nsims)]

            self._yobs = [self._yobs_view[n].reshape(
                len(tout[n]) * len(obs_names)).view(dtype=yobs_dtype) for n
                          in range(self.nsims)]
            self._yexpr = [self._yexpr_view[n].reshape(
                len(tout[n]) * len(expr_names)).view(dtype=yexpr_dtype) for n
                          in range(self.nsims)]
        else:
            self._yobs = [np.ndarray((len(self.tout[n]),),
                                     dtype=yobs_dtype) for n in range(self.nsims)]
            self._yobs_view = [self._yobs[n].view(float).
                               reshape(len(self._yobs[n]), -1) for n in range(
                self.nsims)]
            self._yexpr = [np.ndarray((len(self.tout[n]),),
                                      dtype=yexpr_dtype) for n in range(
                self.nsims)]
            self._yexpr_view = [self._yexpr[n].view(float).reshape(len(
                self._yexpr[n]), -1) for n in range(self.nsims)]

            # loop over simulations
            sym_names = obs_names + param_names
            expanded_exprs = [sympy.lambdify(sym_names, expr.expand_expr(),
                                             "numpy") for expr in exprs]
            for n in range(self.nsims):
                if simulator:
                    simulator._logger.log(EXTENDED_DEBUG,
                                          'Evaluating exprs/obs %d/%d'
                                          % (n + 1, self.nsims))

                # observables
                for i, obs in enumerate(model_obs):
                    self._yobs_view[n][:, i] = (
                        self._y[n][:, obs.species] * obs.coefficients).sum(axis=1)

                # expressions
                sym_dict = dict((k, self._yobs[n][k]) for k in obs_names)
                sym_dict.update(dict((p.name, self.param_values[
                    n // self.n_sims_per_parameter_set][i]) for i, p in
                                enumerate(self._model.parameters)))
                for i, expr in enumerate(exprs):
                    self._yexpr_view[n][:, i] = expanded_exprs[i](**sym_dict)

        if simulator:
            simulator._reset_run_overrides()
            simulator._logger.debug('SimulationResult constructor finished')

    def _squeeze_output(self, trajectories):
        """
        Reduces trajectories to a 2D matrix if only one simulation present

        Can be disabled by setting self.squeeze to False
        """
        if self.nsims == 1 and self.squeeze:
            return trajectories[0]
        else:
            return trajectories

    @property
    def nsims(self):
        """ The number of simulations in this SimulationResult """
        return self._nsims

    @property
    def all(self):
        """
        Aggregate species, observables, and expressions trajectories into
        a numpy.ndarray with record-style data-type for return to the user.
        """
        if self._yfull is None:
            sp_names = ['__s%d' % i for i in range(len(self._model.species))]
            yfull_dtype = list(zip(sp_names, itertools.repeat(float)))
            if len(self._model.observables):
                yfull_dtype += self._yobs[0].dtype.descr
            if len(self._model.expressions_dynamic()):
                yfull_dtype += self._yexpr[0].dtype.descr
            yfull = []
            # loop over simulations
            for n in range(self.nsims):
                yfull.append(np.ndarray(len(self.tout[n]), yfull_dtype))
                yfull_view = yfull[n].view(float).reshape((len(yfull[n]), -1))
                n_sp = self._y[n].shape[1] if self._y else 0
                n_ob = self._yobs_view[n].shape[1]
                n_ex = self._yexpr_view[n].shape[1]
                if self._y:
                    yfull_view[:, :n_sp] = self._y[n]
                yfull_view[:, n_sp:n_sp + n_ob] = self._yobs_view[n]
                yfull_view[:, n_sp + n_ob:n_sp + n_ob + n_ex] = \
                    self._yexpr_view[n]
            self._yfull = yfull

        return self._squeeze_output(self._yfull)

    @property
    def dataframe(self):
        """
        A conversion of the trajectory sets (species, observables and
        expressions for all simulations) into a single
        :py:class:`pandas.DataFrame`.
        """
        if pd is None:
            raise Exception('Please "pip install pandas" for this feature')
        sim_ids = (np.repeat(range(self.nsims), [len(t) for t in self.tout]))
        times = np.concatenate(self.tout)
        if self.nsims == 1 and self.squeeze:
            idx = pd.Index(times, name='time')
        else:
            idx = pd.MultiIndex.from_tuples(list(zip(sim_ids, times)),
                                            names=['simulation', 'time'])
        simdata = self.all
        if not isinstance(simdata, np.ndarray):
            simdata = np.concatenate(simdata)
        return pd.DataFrame(simdata, index=idx)

    @property
    def species(self):
        """
        List of trajectory sets. The first dimension contains species.
        """
        if self._y is None:
            raise ValueError('No trajectories are available for network-free '
                             'simulations')
        return self._squeeze_output(self._y)

    @property
    def observables(self):
        """
        List of trajectory sets. The first dimension contains observables.
        """
        if not self._model.observables:
            raise ValueError('Model has no observables')
        return self._squeeze_output(self._yobs)

    def observable(self, pattern):
        """
        Calculate a pattern's trajectories without adding to model

        This method calculates an observable "on demand" using
        any supplied MonomerPattern or ComplexPattern against the simulation
        result, without re-running the simulation.

        Note that the monomers within the supplied pattern are reconciled
        with the SimulationResult's internal copy of the model by name. This
        method only works on simulations which calculate species
        trajectories (i.e. it will not work on network-free simulations).

        Raises a ValueError if the pattern does not match at least one species.

        Parameters
        ----------
        pattern: pysb.MonomerPattern or pysb.ComplexPattern
            An observable pattern to match

        Returns
        -------
        pandas.Series
            Series containing the simulation trajectories for the specified
            observable

        Examples
        --------

        >>> from pysb import ANY
        >>> from pysb.examples import earm_1_0
        >>> from pysb.simulator import ScipyOdeSimulator
        >>> simres = ScipyOdeSimulator(earm_1_0.model, tspan=range(5)).run()
        >>> m = earm_1_0.model.monomers

        Observable of bound Bid:

        >>> simres.observable(m.Bid(b=ANY))
        time
        0    0.000000e+00
        1    1.190933e-12
        2    2.768582e-11
        3    1.609716e-10
        4    5.320530e-10
        dtype: float64

        Observable of AMito bound to mCytoC:

        >>> simres.observable(m.AMito(b=1) % m.mCytoC(b=1))
        time
        0    0.000000e+00
        1    1.477319e-77
        2    1.669917e-71
        3    5.076939e-69
        4    1.157400e-66
        dtype: float64
        """

        # Adjust the supplied pattern's monomer objects to match the
        # simulationresult's internal model
        if isinstance(pattern, MonomerPattern):
            self._update_monomer_pattern(pattern)
        elif isinstance(pattern, ComplexPattern):
            for mp in pattern.monomer_patterns:
                self._update_monomer_pattern(mp)
        else:
            raise ValueError('The pattern must be a MonomerPattern or '
                             'ComplexPattern')

        if self._y is None:
            raise ValueError('On demand observables can only be calculated '
                             'on simulations with species trajectories')

        obs_matches = SpeciesPatternMatcher(self._model).match(
            pattern, index=True, counts=True)

        if not obs_matches:
            raise ValueError('No species match the supplied observable '
                             'pattern')

        return self.dataframe.iloc[:, list(obs_matches.keys())].multiply(
            list(obs_matches.values())).sum(axis=1)

    def _update_monomer_pattern(self, pattern):
        """ Update a pattern's monomer objects to use internal model

        Internal function for in-place update of a pattern to replace its
        monomers with those from SimulationResult's model, matching by name.

        Raises ValueError if no monomer with the specified name is in the
        model.
        """
        mon_name = pattern.monomer.name
        try:
            new_mon = self._model.monomers[mon_name]
        except KeyError:
            raise ValueError('There was no monomer called "{}" in the model '
                             '"{}" at the time of simulation'.format(
                                mon_name, self._model.name))
        pattern.monomer = new_mon

    @property
    def expressions(self):
        """
        List of trajectory sets. The first dimension contains expressions.
        """
        if not self._model.expressions_dynamic():
            raise ValueError('Model has no dynamic expressions')
        return self._squeeze_output(self._yexpr)

    @property
    def initials(self):
        return self._initials

    @property
    def param_values(self):
        return self._param_values

    def save(self, filename, dataset_name=None, group_name=None,
             append=False, include_obs_exprs=False):
        """
        Save a SimulationResult to a file (HDF5 format)

        HDF5 is a hierarchical, binary storage format well suited to storing
        matrix-like data. Our implementation requires the h5py package.

        Each SimulationResult is treated as an HDF5 dataset, stored within a
        group which is specific to a model. In this way, it is possible to save
        multiple SimulationResults for a specific model.

        A group is first created in the HDF file root (see group_name
        argument). Within that group, a dataset "_model" has a pickled
        version of the PySB model. SimulationResult are stored as groups
        within the model group.

        The file hierarchy under group_name/dataset_name/ then consists of
        the following HDF5 gzip compressed HDF5 datasets: trajectories,
        param_values, initials, tout, observables (optional) and expressions
        (optional); and the following attributes:
        simulator_class (pickled Class), simulator_kwargs (pickled dict),
        squeeze (bool), simulations_per_param_set (int), pysb_version (str),
        timestamp (ISO 8601 format).

        Custom attributes can be stored in the SimulationResult's
        `custom_attrs` dictionary. Keys should be strings, values can be any
        picklable object. When saved to HDF5, these custom attributes will
        be prefixed with ``usrattr_``.

        Parameters
        ----------
        filename: str
            Filename to which the data will be saved
        dataset_name: str or None
            Dataset name. If None, it will default to 'result'. If the
            dataset_name already exists within the group, a ValueError is
            raised.
        group_name: str or None
            Group name. If None, will default to the name of the model.
        append: bool
            If False, raise IOError if the specified file already exists. If
            True, append to existing file (or create if it doesn't exist).
        include_obs_exprs: bool
            Whether to save observables and expressions in the file or not.
            If they are not included, they can be recreated from the model
            and species trajectories when loaded back into PySB, but you may
            wish to include them for use with external software, or if you
            have complex expressions which take a long time to compute.

        """
        if h5py is None:
            raise Exception('Please install the h5py package for this feature')

        if self._y is None and not include_obs_exprs:
            warn('This SimulationResult has no trajectories - '
                 'you will need to set include_obs_exprs=True if '
                 'you wish to save observables and expressions')

        if group_name is None:
            group_name = self._model.name
        if dataset_name is None:
            dataset_name = 'result'

        # np.void maps to bytes in HDF5.
        enpickle = lambda obj: np.void(pickle.dumps(obj, -1))

        with h5py.File(filename, 'a' if append else 'w-') as hdf:
            # Get or create the group
            try:
                grp = hdf.create_group(group_name)
                grp.create_dataset('_model', data=enpickle(self._model))
            except ValueError:
                grp = hdf[group_name]
                model = pickle.loads(grp['_model'][()])
                if model.name != self._model.name:
                    raise ValueError('SimulationResult model has name "{}", '
                                     'but the model in HDF5 file group "{}" '
                                     'has name "{}"'.format(self._model.name,
                                                            group_name,
                                                            model.name))

            # Create the result dataset, which is actually a nested HDF group
            dset = grp.create_group(dataset_name)
            if self._y is not None:
                dset.create_dataset('trajectories', data=self._y,
                                    compression='gzip', shuffle=True)
            if include_obs_exprs:
                dset.create_dataset('observables', data=self._yobs_view,
                                    compression='gzip', shuffle=True)
                dset.create_dataset('expressions', data=self._yexpr_view,
                                    compression='gzip', shuffle=True)
            dset.create_dataset('param_values', data=self.param_values,
                                compression='gzip', shuffle=True)
            if isinstance(self.initials, np.ndarray):
                dset.create_dataset('initials', data=self.initials,
                                    compression='gzip', shuffle=True)
            else:
                dset.create_dataset('initials_dict', data=enpickle(
                    self.initials))
            dset.create_dataset('tout', data=self.tout,
                                compression='gzip')
            dset.attrs['simulator_class'] = enpickle(self.simulator_class)
            dset.attrs['init_kwargs'] = enpickle(self.init_kwargs)
            dset.attrs['run_kwargs'] = enpickle(self.run_kwargs)
            dset.attrs['squeeze'] = self.squeeze
            dset.attrs['simulations_per_param_set'] = \
                self.n_sims_per_parameter_set
            dset.attrs['pysb_version'] = self.pysb_version
            dset.attrs['timestamp'] = datetime.isoformat(
                self.timestamp)
            # This is the range of ints that can be natively encoded in HDF5.
            int_min = np.iinfo(np.int64).min
            int_max = np.iinfo(np.uint64).max
            for attr_name, attr_val in self.custom_attrs.items():
                # Pass HDF5-native values straight through, pickling others.
                if (not (isinstance(attr_val,
                                    (basestring, bytes, float, complex))
                         or (isinstance(attr_val, numbers.Integral)
                             and int_min <= attr_val <= int_max))):
                    attr_val = enpickle(attr_val)
                dset.attrs[self.CUSTOM_ATTR_PREFIX + attr_name] = attr_val

    @classmethod
    def load(cls, filename, dataset_name=None, group_name=None):
        """
        Load a SimulationResult from a file (HDF5 format)

        For a description of the file format see :func:`save`

        Parameters
        ----------
        filename: str
            Filename from which to load data
        dataset_name: str or None
            Dataset name. Can be left as None when the group specified only
            contains one dataset, which will then be selected. If None and
            more than one dataset is in the group, a ValueError is raised.
        group_name: str or None
            Group name. This is typically the name of the model. Can be left as
            None when the file only contains one group, which will then be
            selected. If None and more than group is in the file a
            ValueError is raised.

        Returns
        -------
        SimulationResult
            Set of trajectories and associated metadata loaded from the file
        """
        if h5py is None:
            raise Exception('Please "pip install h5py" for this feature')

        with h5py.File(filename, 'r') as hdf:
            if group_name is None:
                groups = hdf.keys()
                if len(groups) > 1:
                    raise ValueError("group_name must be specified when file "
                                     "contains more than one group. Options "
                                     "are: {}".format(str(groups)))
                group_name = next(iter(hdf))

            grp = hdf[group_name]

            if dataset_name is None:
                datasets = list(grp.keys())
                datasets.remove('_model')
                if len(datasets) > 1:
                    raise ValueError("dataset_name must be specified when "
                                     "group contains more than one dataset. "
                                     "Options are: {}".format(str(datasets)))
                dataset_name = datasets[0]

            dset = grp[dataset_name]

            obs_and_exprs = None

            if 'observables' in dset.keys():
                obs_and_exprs = list(dset['observables'])

            if 'expressions' in dset.keys():
                exprs = dset['expressions']
                if obs_and_exprs is None:
                    obs_and_exprs = list(exprs)
                else:
                    for i in range(len(obs_and_exprs)):
                        obs_and_exprs[i] = np.concatenate(
                            [obs_and_exprs[i], exprs[i]],
                            axis=1
                        )

            trajectories = None
            try:
                trajectories = dset['trajectories']
            except KeyError:
                pass

            try:
                initials = np.array(dset['initials'])
            except KeyError:
                initials = pickle.loads(dset['initials_dict'][()])

            simres = cls(
                simulator=None,
                model=pickle.loads(grp['_model'][()]),
                initials=initials,
                param_values=np.array(dset['param_values']),
                tout=np.array(dset['tout']),
                trajectories=trajectories,
                observables_and_expressions=obs_and_exprs,
                squeeze=dset.attrs['squeeze'],
                simulations_per_param_set=dset.attrs[
                    'simulations_per_param_set']
            )
            simres.pysb_version = dset.attrs['pysb_version']
            simres.timestamp = dateutil.parser.parse(
                dset.attrs['timestamp'])
            simres.simulator_class = pickle.loads(
                dset.attrs['simulator_class'])
            simres.init_kwargs = pickle.loads(dset.attrs['init_kwargs'])
            simres.run_kwargs = pickle.loads(dset.attrs['run_kwargs'])
            for attr_name in dset.attrs.keys():
                if attr_name.startswith(cls.CUSTOM_ATTR_PREFIX):
                    orig_name = attr_name[len(cls.CUSTOM_ATTR_PREFIX):]
                    attr_val = dset.attrs[attr_name]
                    # Restore objects that were pickled for storage.
                    if isinstance(attr_val, np.void):
                        attr_val = pickle.loads(attr_val)
                    simres.custom_attrs[orig_name] = attr_val
            return simres


def _allow_unicode_recarray():
    """Return True if numpy recarray can take unicode data type.

    In python 2, numpy doesn't allow unicode strings as names in arrays even
    if they are ascii encodeable. This function tests this directly.
    """
    try:
        np.ndarray((1,), dtype=[(u'X', float)])
    except TypeError:
        return False
    return True
