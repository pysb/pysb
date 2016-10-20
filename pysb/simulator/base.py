from abc import ABCMeta, abstractmethod
import numpy as np
import itertools
import sympy
import collections
from pysb.core import MonomerPattern, ComplexPattern, as_complex_pattern, \
                      Component
try:
    import pandas as pd
except ImportError:
    pd = None


class SimulatorException(Exception):
    pass


class Simulator(object):
    """An abstract base class for numerical simulation of models.

    Please note that the interface for this class is considered
    experimental and may change without warning as PySB is updated.

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
    verbose : bool, optional (default: False)
        Verbose output.

    Attributes
    ----------
    verbose: bool
        Verbosity flag passed to the constructor.
    model : pysb.Model
        Model passed to the constructor.
    tspan : vector-like
        Time values passed to the constructor.
    tout: numpy.ndarray
        Time points returned by the simulator (may be different from ``tspan``
        if simulation is interrupted for some reason).

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
                    if not isinstance(val, collections.Iterable):
                        new_initials[cplx_pat] = np.array([val])
                    # otherwise, check whether simulator supports multiple 
                    # initial values
                    elif len(val) > 1 and not self._supports['multi_initials']:
                        raise SimulatorException(self.__class__.__name__ + 
                                        " does not support multiple initial"
                                        " values at this time.")
                self._initials = new_initials
            else:
                if not isinstance(new_initials, np.ndarray):
                    new_initials = np.array(new_initials, copy=False)
                # if new_initials is a 1D array, convert to a 2D array of length 1
                if len(new_initials.shape) == 1:
                    new_initials = np.resize(new_initials, (1,len(new_initials)))
                # check whether simulator supports multiple initial values
                elif not self._supports['multi_initials']:
                    raise SimulatorException(self.__class__.__name__ + 
                                    " does not support multiple initial"
                                    " values at this time.")
                # make sure number of initials values equals len(model.species)
                if new_initials.shape[1] != len(self._model.species):
                    raise ValueError("new_initials must be the same length as "
                                     "model.species")
                self._initials = new_initials

#     @property
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
        n_sims = 1
        if isinstance(self._initials, dict):
            # record the length of the arrays and make 
            # sure they're all the same.
            for key,val in self._initials.items():                    
                if n_sims == 1:
                    n_sims = len(val)
                elif len(val) != n_sims:
                    raise Exception("all arrays in new_initials dictionary "
                                    "must be equal length")
        else:
            self._initials = {}
        y0 = np.zeros((len(self._model.species),))
        y0 = np.repeat([y0], n_sims, axis=0)
        # note that param_vals is a 2D array
        subs = [dict((p, pv[i]) for i, p in
                    enumerate(self._model.parameters)) for pv in self.param_values]

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
                    if y0[sim][si] != 0:
                        continue
                    
                    def _get_value(sim):
                        if isinstance(value_obj, collections.Iterable) and \
                           isinstance(value_obj[sim], (int, float)):
                            value = value_obj[sim]
                        elif isinstance(value_obj, Component):
                            if value_obj in self._model.parameters:
                                pi = self._model.parameters.index(value_obj)
                                value = self.param_values[sim][pi]
                            elif value_obj in self._model.expressions:
                                value = value_obj.expand_expr().evalf(subs=subs[sim])
                        else:
                            raise TypeError("Unexpected initial condition value type")
                        return value
                        
                    # initials from the model
                    if isinstance(initials_source, np.ndarray):
                        if len(initials_source.shape) == 1:
                            if sim == 0:
                                value = _get_value(0)
                            else:
                                # if the parameters are different for each sim, the expressions
                                # could be different too
                                if value_obj in self._model.expressions:
                                    value = value_obj.expand_expr().evalf(subs=subs[sim])
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
                for key,val in param_values_dict.items():                    
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
            for key,val in new_params.items():
                if key not in self._model.parameters.keys():
                    raise IndexError("new_params dictionary has unknown "
                                     "parameter name (%s)" % key)
                # if val is a number, convert it to a single-element array
                if not isinstance(val, collections.Iterable):
                    new_params[key] = np.array([val])
                # otherwise, check whether simulator supports multiple 
                # param_values
                elif len(val) > 1 and not self._supports['multi_param_values']:
                    raise SimulatorException(self.__class__.__name__ + 
                                    " does not support multiple parameter"
                                    " values at this time.")
                    # NOTE: Strings are iterables, so they fall here
                    #       Should we catch strings explicitly?
            self._params = new_params
        else:
            if not isinstance(new_params, np.ndarray):
                new_params = np.array(new_params)
            # if new_params is a 1D array, convert to a 2D array of length 1
            if len(new_params.shape) == 1:
                new_params = np.resize(new_params, (1,len(new_params)))
            # check whether simulator supports multiple parameter values
            elif not self._supports['multi_param_values']:
                raise SimulatorException(self.__class__.__name__ + 
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
        return None


class SimulationResult(object):
    """
    Results of a simulation with properties and methods to access them.

    Please note that the interface for this class is considered
    experimental and may change without warning as PySB is updated.

    Parameters
    ----------
    simulator : Simulator
        The simulator object that generated the trajectories
    trajectories : list or numpy.ndarray
        A set of species trajectories from a simulation. Should either be a
        list of 2D numpy arrays or a single 3D numpy array.

    Attributes
    ----------
    In the descriptions below, a "trajectory set" is a 2D numpy array,
    species on first axis and time on second axis, with each element
    containing the concentration or count of the species at the specified time.

    A list of trajectory sets contains a trajectory set for each simulation.

    all : list
        List of trajectory sets. The first dimension contains species,
        observables and expressions (in that order)
    species : list
        List of trajectory sets. The first dimension contains species.
    observables : list
        List of trajectory sets. The first dimension contains observables.
    expressions : list
        List of trajectory sets. The first dimension contains expressions.
    dataframe : :py:class:`pandas.DataFrame`
        A conversion of the trajectory sets (species, observables and
        expressions for all simulations) into a single
        :py:class:`pandas.DataFrame`.
    """
    def __init__(self, simulator, trajectories):
        self.squeeze = True
        self.simulator = type(simulator).__name__
        self.tout = simulator.tout
        self._yfull = None
        self._model = simulator._model

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
        
        self.nsims = len(self._y)
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
        obs_names = model_obs.keys()
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
        for n in range(self.nsims):
            # observables
            for i, obs in enumerate(model_obs):
                self._yobs_view[n][:, i] = (
                    self._y[n][:, obs.species] * obs.coefficients).sum(axis=1)

            # expressions
            obs_dict = dict((k, self._yobs[n][k]) for k in obs_names)
            subs = dict((p, param_values[n][i]) for i, p in
                        enumerate(self._model.parameters))
            for i, expr in enumerate(exprs):
                expr_subs = expr.expand_expr().subs(subs)
                func = sympy.lambdify(obs_names, expr_subs, "numpy")
                self._yexpr_view[n][:, i] = func(**obs_dict)

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
    def all(self):
        """Aggregate species, observables, and expressions trajectories into
        a numpy.ndarray with record-style data-type for return to the user."""
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
        return self._squeeze_output(self._y)

    @property
    def observables(self):
        return self._squeeze_output(self._yobs)

    @property
    def expressions(self):
        return self._squeeze_output(self._yexpr)
