from abc import ABCMeta, abstractmethod
import numpy as np
import itertools
import sympy
import collections
from pysb.core import MonomerPattern, ComplexPattern, as_complex_pattern
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
        if self._initials is not None:
            return self._initials
        else:
            return self.initials_list

    @initials.setter
    def initials(self, new_initials):
        if new_initials is None:
            self._initials = None
        else:
            # check if y0 is a Mapping, and if so validate the keys
            # (ComplexPatterns)
            if isinstance(new_initials, collections.Mapping):
                for cplx_pat, val in new_initials.items():
                    if not isinstance(cplx_pat, (MonomerPattern,
                                                 ComplexPattern)):
                        raise SimulatorException('Dictionary key %s is not a '
                                                 'MonomerPattern or '
                                                 'ComplexPattern' %
                                                 repr(cplx_pat))
                self._initials = new_initials
            # accept vector of species amounts as an argument
            elif len(new_initials) != len(self._model.species):
                raise ValueError("new_initials must be the same length as "
                                 "model.species")
            else:
                self._initials = np.array(new_initials, copy=False)

    @property
    def initials_list(self):
        """
        Returns the model's initial conditions as a list, with the order
        matching model.initial_conditions
        """
        # If we already have a list internally, just return that
        if isinstance(self._initials, np.ndarray):
            return self._initials
        # Otherwise, build the list from the model, and any overrides
        # specified in the self._initials dictionary
        y0 = np.zeros((len(self._model.species),))
        param_vals = self.param_values
        subs = dict((p, param_vals[i]) for i, p in
                    enumerate(self._model.parameters))

        def _set_initials(initials_source):
            for cp, value_obj in initials_source:
                cp = as_complex_pattern(cp)
                si = self._model.get_species_index(cp)
                if si is None:
                    raise IndexError("Species not found in model: %s" %
                                     repr(cp))
                # If this initial condition has already been set, skip it
                if y0[si] != 0:
                    continue
                if isinstance(value_obj, (int, float)):
                    value = value_obj
                elif value_obj in self._model.parameters:
                    pi = self._model.parameters.index(value_obj)
                    value = param_vals[pi]
                elif value_obj in self._model.expressions:
                    value = value_obj.expand_expr().evalf(subs=subs)
                else:
                    raise ValueError("Unexpected initial condition value type")
                y0[si] = value

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
            param_values_dict = self._params if isinstance(self._params,
                                                           dict) else {}
            param_values = np.array([p.value for p in self._model.parameters])
            for key in param_values_dict.keys():
                try:
                    pi = self._model.parameters.index(self._model.parameters[
                                                         key])
                except KeyError:
                    raise IndexError("new_params dictionary has unknown "
                                     "parameter name (%s)" % key)
                param_values[pi] = param_values_dict[key]
            return param_values

    @param_values.setter
    def param_values(self, new_params):
        if new_params is None:
            self._params = None
            return
        if isinstance(new_params, dict):
            for k in new_params.keys():
                if k not in self._model.parameters.keys():
                    raise IndexError("new_params dictionary has unknown "
                                     "parameter name (%s)" % k)
            self._params = new_params
        else:
            # accept vector of parameter values as an argument
            if len(new_params) != len(self._model.parameters):
                raise ValueError("new_params must be the same length as "
                                 "model.parameters")
            if isinstance(new_params, np.ndarray):
                self._params = new_params
            else:
                self._params = np.array(new_params)

    @abstractmethod
    def run(self, tspan=None, param_values=None, initials=None):
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
            subs = dict((p, param_values[i]) for i, p in
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
