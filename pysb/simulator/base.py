from abc import ABCMeta, abstractmethod
import numpy as np
import itertools
import sympy
import collections
import numbers
from pysb.core import MonomerPattern, ComplexPattern, as_complex_pattern, \
                      Component
from pysb.logging import get_logger, EXTENDED_DEBUG
import logging

try:
    import pandas as pd
except ImportError:
    pd = None


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
        self.tspan = tspan
        self._initials = None
        self.initials = initials
        self._params = None
        self.param_values = param_values

    @property
    def model(self):
        return self._model

    @property
    def initials(self):
        return self._initials if isinstance(self._initials, np.ndarray) \
               else self._get_initials()

    @initials.setter
    def initials(self, new_initials):
        if new_initials is None:
            self._initials = None
        else:
            # check if new_initials is a Mapping, and if so validate the keys
            # (ComplexPatterns)
            if isinstance(new_initials, collections.Mapping):
                for cplx_pat, val in new_initials.items():
                    if not isinstance(cplx_pat, (MonomerPattern,
                                                 ComplexPattern)):
                        raise SimulatorException('Dictionary key %s is not a '
                                                 'MonomerPattern or '
                                                 'ComplexPattern' %
                                                 repr(cplx_pat))
                    # if val is a number, convert it to a single-element array
                    if not isinstance(val, (collections.Sequence, np.ndarray)):
                        val = [val]
                        new_initials[cplx_pat] = np.array(val)
                    # otherwise, check whether simulator supports multiple
                    # initial values
                    elif len(val) > 1:
                        if not self._supports['multi_initials']:
                            raise SimulatorException(
                                self.__class__.__name__ +
                                " does not support multiple initial"
                                " values at this time.")
                        if 1 < len(self.param_values) != len(val):
                            raise SimulatorException(
                                'Cannot set initials for {} simulations '
                                'when param_values has been set for {} '
                                'simulations'.format(
                                    len(val), len(self.param_values)))
                    if not np.isfinite(val).all():
                        raise SimulatorException('Please check initial {} '
                                                 'for non-finite '
                                                 'values'.format(cplx_pat))
                self._initials = new_initials
            else:
                if not isinstance(new_initials, np.ndarray):
                    new_initials = np.array(new_initials, copy=False)
                # if new_initials is a 1D array, convert to a 2D array of length 1
                if len(new_initials.shape) == 1:
                    new_initials = np.resize(new_initials, (1,len(new_initials)))
                # check whether simulator supports multiple initial values
                else:
                    if not self._supports['multi_initials']:
                        raise SimulatorException(
                            self.__class__.__name__ +
                            " does not support multiple initial"
                            " values at this time.")
                    if 1 < len(self.param_values) != new_initials.shape[0]:
                        raise SimulatorException(
                            'Cannot set initials for {} simulations '
                            'when param_values has been set for {} '
                            'simulations'.format(
                                new_initials.shape[0], len(self.param_values)))
                # make sure number of initials values equals len(model.species)
                if new_initials.shape[1] != len(self._model.species):
                    raise ValueError("new_initials must be the same length as "
                                     "model.species")
                if not np.isfinite(new_initials).all():
                    raise SimulatorException('Please check initials array'
                                             'for non-finite values')
                self._initials = new_initials

    def _get_initials(self):
        """
        Returns the model's initial conditions, with the order
        matching model.initial_conditions
        """
        # If we already have a list internally, just return that
        if isinstance(self._initials, np.ndarray):
            return self._initials
        # Otherwise, build the list from the model, and any overrides
        # specified in the self._initials dictionary
        if isinstance(self._initials, dict):
            try:
                n_sims_initials = len(self._initials.values()[0])
            except IndexError:
                n_sims_initials = 1
            # record the length of the arrays and make
            # sure they're all the same.
            for key, val in self._initials.items():
                if len(val) != n_sims_initials:
                    raise Exception("all arrays in new_initials dictionary "
                                    "must be equal length")
        else:
            n_sims_initials = 1
            self._initials = {}
        n_sims_params = len(self.param_values)
        n_sims_actual = max(n_sims_params, n_sims_initials)

        y0 = np.full((n_sims_actual, len(self.model.species)), np.nan)

        # note that param_vals is a 2D array
        subs = [dict((p, pv[i]) for i, p in enumerate(self._model.parameters))
                for pv in self.param_values]
        if len(subs) == 1 and n_sims_initials > 1:
            subs = list(itertools.repeat(subs[0], n_sims_initials))

        def _set_initials(initials_source):
            for cp, value_obj in initials_source:
                cp = as_complex_pattern(cp)
                si = self._model.get_species_index(cp)
                if si is None:
                    raise IndexError("Species not found in model: %s" %
                                     repr(cp))
                # Loop over all simulations
                for sim in range(len(y0)):
                    # If this initial condition has already been set, skip it
                    # (i.e., an override)
                    if not np.isnan(y0[sim][si]):
                        continue

                    def _get_value(sim):
                        if isinstance(value_obj, (collections.Sequence,
                                                  np.ndarray)) and \
                           isinstance(value_obj[sim], numbers.Number):
                            value = value_obj[sim]
                        elif isinstance(value_obj, Component):
                            if value_obj in self._model.parameters:
                                pi = self._model.parameters.index(value_obj)
                                value = self.param_values[
                                    sim if n_sims_params > 1 else 0][pi]
                            elif value_obj in self._model.expressions:
                                value = value_obj.expand_expr().evalf(subs=subs[sim])
                        else:
                            raise TypeError("Unexpected initial condition "
                                            "value type: %s" % type(value_obj))
                        return value

                    # initials from the model
                    if isinstance(initials_source, np.ndarray):
                        if len(initials_source.shape) == 1:
                            if sim == 0:
                                value = _get_value(0)
                            else:
                                # if the parameters are different for each sim,
                                # the expressions could be different too
                                if value_obj in self._model.expressions:
                                    value = value_obj.expand_expr().evalf(
                                        subs=subs[sim])
                                else:
                                    value = y0[sim-1][si]
                    # initials from dict
                    else:
                         value = _get_value(sim)
                    y0[sim][si] = value

        # Process any overrides
        if isinstance(self._initials, dict):
            _set_initials(self._initials.items())
        # Get remaining initials from the model itself
        _set_initials(self._model.initial_conditions)

        # Any remaining unset initials should be set to zero
        y0 = np.nan_to_num(y0)

        return y0

    @property
    def param_values(self):
        if self._params is not None and not isinstance(self._params, dict):
            return self._params
        else:
            # create parameter vector from the values in the model
            n_sims = 1
            if isinstance(self._params, dict):
                param_values_dict = self._params
                # record the length of the arrays and make
                # sure they're all the same.
                for key, val in param_values_dict.items():
                    if n_sims == 1:
                        n_sims = len(val)
                    elif len(val) != n_sims:
                        raise Exception("all arrays in new_params dictionary "
                                        "must be equal length")
            else:
                param_values_dict = {}
            param_values = np.array([p.value for p in self._model.parameters])
            param_values = np.repeat([param_values], n_sims, axis=0)
            # overrides
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
        if new_params is None:
            self._params = None
            return
        if isinstance(new_params, dict):
            for key, val in new_params.items():
                if key not in self._model.parameters.keys():
                    raise IndexError("new_params dictionary has unknown "
                                     "parameter name (%s)" % key)
                # if val is a number, convert it to a single-element array
                if not isinstance(val, collections.Sequence):
                    new_params[key] = np.array([val])
                # otherwise, check whether simulator supports multiple
                # param_values
                elif len(val) > 1 and not self._supports['multi_param_values']:
                    raise SimulatorException(
                        self.__class__.__name__ +
                        " does not support multiple parameter"
                        " values at this time.")

            self._params = new_params
        else:
            if not isinstance(new_params, np.ndarray):
                new_params = np.array(new_params)
            # if new_params is a 1D array, convert to a 2D array of length 1
            if len(new_params.shape) == 1:
                new_params = np.resize(new_params, (1,len(new_params)))
            # check whether simulator supports multiple parameter values
            elif not self._supports['multi_param_values']:
                raise SimulatorException(
                    self.__class__.__name__ +
                    " does not support multiple parameter"
                    " values at this time.")
            # make sure number of param values equals len(model.parameters)
            if new_params.shape[1] != len(self._model.parameters):
                raise ValueError("new_params must be the same length as "
                                 "model.parameters")
            self._params = new_params

    @abstractmethod
    def run(self, tspan=None, initials=None, param_values=None):
        """Run a simulation.

        Implementations should return a :class:`.SimulationResult` object.
        """
        self._logger.info('Simulation(s) started')
        if tspan is not None:
            self.tspan = tspan
        if self.tspan is None:
            raise SimulatorException("tspan must be defined before "
                                     "simulation can run")
        if param_values is not None:
            self.param_values = param_values
        if initials is not None:
            self.initials = initials
        # If only one set of param_values, run all simulations
        # with the same parameters
        if len(self.param_values) == 1:
            self.param_values = np.repeat(self.param_values,
                                          len(self.initials),
                                          axis=0)

        assert len(self.initials) == len(self.param_values)

        # Error checks on 'param_values' and 'initials'
        if len(self.param_values) != len(self.initials):
            raise SimulatorException(
                    "'param_values' and 'initials' must be equal lengths.\n"
                    "len(param_values): %d\n"
                    "len(initials): %d" %
                    (len(self.param_values), len(self.initials)))
        elif len(self.param_values.shape) != 2 or \
                self.param_values.shape[1] != len(self._model.parameters):
            raise SimulatorException(
                    "'param_values' must be a 2D array of dimension N_SIMS x "
                    "len(model.parameters).\n"
                    "param_values.shape: " + str(self.param_values.shape) +
                    "\nlen(model.parameters): %d" %
                    len(self._model.parameters))
        elif len(self.initials.shape) != 2 or \
                self.initials.shape[1] != len(self._model.species):
            raise SimulatorException(
                    "'initials' must be a 2D array of dimension N_SIMS x "
                    "len(model.species).\n"
                    "initials.shape: " + str(self.initials.shape) +
                    "\nlen(model.species): %d" % len(self._model.species))
            # NOTE: Not sure if the check on 'initials' should be here or not.
            # Network-free simulators don't have species, but we will want to
            # allow users to supply initials. Right now, that will raise an
            # exception. Need to think about this. --LAH
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
    [  1.0000e+00   1.1744e-02   1.3791e-04   1.6196e-06   1.9020e-08
       2.2337e-10   2.6232e-12   3.0806e-14   3.6178e-16   4.2492e-18]

    It is also possible to retrieve the value of all observables at a
    particular time point, e.g. the final concentrations:

    >>> print(simulation_result.observables[-1]) \
        #doctest: +NORMALIZE_WHITESPACE
    (  4.2492e-18,   1.6996e-16,  1.)

    Expressions are read in the same way as observables:

    >>> print(simulation_result.expressions['NBD_signal']) \
        #doctest: +NORMALIZE_WHITESPACE
    [ 0.   4.7847  4.9956  4.9999  5.   5.   5.   5.   5.   5. ]

    The species trajectories can be accessed as a numpy ndarray:

    >>> print(simulation_result.species)
    [[  1.0000e+00   0.0000e+00   0.0000e+00]
     [  1.1744e-02   5.2194e-02   9.3606e-01]
     [  1.3791e-04   1.2259e-03   9.9864e-01]
     [  1.6196e-06   2.1595e-05   9.9998e-01]
     [  1.9020e-08   3.3814e-07   1.0000e+00]
     [  2.2337e-10   4.9637e-09   1.0000e+00]
     [  2.6232e-12   6.9951e-11   1.0000e+00]
     [  3.0806e-14   9.5840e-13   1.0000e+00]
     [  3.6178e-16   1.2863e-14   1.0000e+00]
     [  4.2492e-18   1.6996e-16   1.0000e+00]]

    Species, observables and expressions can be combined into a single numpy
    ndarray and accessed similarly. Here, the initial concentrations of all
    these entities are examined:

    >>> print(simulation_result.all[0]) #doctest: +NORMALIZE_WHITESPACE
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
    def __init__(self, simulator, tout, trajectories, squeeze=True,
                 simulations_per_param_set=1):
        simulator._logger.debug('SimulationResult constructor started')
        self.squeeze = squeeze
        self.tout = tout
        self._yfull = None
        self._model = simulator._model
        self.n_sims_per_parameter_set = simulations_per_param_set

        # Validate incoming trajectories
        if hasattr(trajectories, 'ndim') and trajectories.ndim == 3:
            # trajectories is a 3D array, create a list of 2D arrays
            # This is just a view and doesn't copy the data
            self._y = [tr for tr in trajectories]
        else:
            # Not a 3D array, check for a list of 2D arrays
            try:
                if any([tr.ndim != 2 for tr in trajectories]):
                    raise AttributeError
            except (AttributeError, TypeError):
                raise ValueError("trajectories should be a 3D array or a list "
                                 "of 2D arrays")
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

        # Calculate ``yobs`` and ``yexpr`` based on values of ``y``
        exprs = self._model.expressions_dynamic()
        expr_names = [expr.name for expr in exprs]
        model_obs = self._model.observables
        obs_names = list(model_obs.keys())

        if not _allow_unicode_recarray():
            for name_list, name_type in zip((expr_names, obs_names),
                                            ('Expression', 'Observable')):
                for i, name in enumerate(name_list):
                    try:
                        name_list[i] = name.encode('ascii')
                    except UnicodeEncodeError:
                        error_msg = 'Non-ASCII compatible' + \
                                    '%s names not allowed' % name_type
                        raise ValueError(error_msg)

        yobs_dtype = zip(obs_names, itertools.repeat(float)) if obs_names \
            else float
        yexpr_dtype = zip(expr_names, itertools.repeat(float)) if expr_names \
            else float

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
        param_values = simulator.param_values

        # loop over simulations
        sym_names = obs_names + [p.name for p in self._model.parameters]
        expanded_exprs = [sympy.lambdify(sym_names, expr.expand_expr(),
                                         "numpy") for expr in exprs]
        for n in range(self.nsims):
            simulator._logger.log(EXTENDED_DEBUG, 'Evaluating exprs/obs %d/%d'
                                  % (n + 1, self.nsims))

            # observables
            for i, obs in enumerate(model_obs):
                self._yobs_view[n][:, i] = (
                    self._y[n][:, obs.species] * obs.coefficients).sum(axis=1)

            # expressions
            sym_dict = dict((k, self._yobs[n][k]) for k in obs_names)
            sym_dict.update(dict((p.name, param_values[
                n // self.n_sims_per_parameter_set][i]) for i, p in
                            enumerate(self._model.parameters)))
            for i, expr in enumerate(exprs):
                self._yexpr_view[n][:, i] = expanded_exprs[i](**sym_dict)

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
            yfull_dtype = zip(sp_names, itertools.repeat(float))
            if len(self._model.observables):
                yfull_dtype += zip(self._model.observables.keys(),
                                   itertools.repeat(float))
            if len(self._model.expressions_dynamic()):
                yfull_dtype += zip(self._model.expressions_dynamic().keys(),
                                   itertools.repeat(float))
            yfull = len(self._y) * [None]
            # loop over simulations
            for n in range(self.nsims):
                yfull[n] = np.ndarray(len(self.tout[n]), yfull_dtype)
                yfull_view = yfull[n].view(float).reshape((len(yfull[n]), -1))
                n_sp = self._y[n].shape[1]
                n_ob = self._yobs_view[n].shape[1]
                n_ex = self._yexpr_view[n].shape[1]
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
            idx = pd.MultiIndex.from_tuples(zip(sim_ids, times),
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
        return self._squeeze_output(self._y)

    @property
    def observables(self):
        """
        List of trajectory sets. The first dimension contains observables.
        """
        return self._squeeze_output(self._yobs)

    @property
    def expressions(self):
        """
        List of trajectory sets. The first dimension contains expressions.
        """
        return self._squeeze_output(self._yexpr)


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
