from abc import ABCMeta, abstractmethod
import numpy as np
import itertools
import sympy
from pysb.core import MonomerPattern, ComplexPattern, as_complex_pattern


class SimulatorException(Exception):
    pass


class Simulator(object):
    """An abstract base class for numerical simulation of models.

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
    def __init__(self, model, verbose=False, **kwargs):
        self.model = model
        self.verbose = verbose
        self.tout = None
        self._y = None
        self._yobs = None
        self._yobs_view = None
        self._yexpr = None
        self._yexpr_view = None
        self._yfull = None
        # Per-run initial conditions/parameter/tspan override
        self.tspan = kwargs.get('tspan', None)
        self._initials = None
        self.initials = kwargs.get('initials', None)
        self._params = None
        self.param_values = kwargs.get('param_values', None)

    def _clear_outputs(self):
        """
        Resets simulation output
        """
        self._y = None
        self._yobs = None
        self._yobs_view = None
        self._yexpr = None
        self._yexpr_view = None
        self._yfull = None

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
            # check if y0 is a dict, and if so validate the keys
            # (ComplexPatterns)
            if isinstance(new_initials, dict):
                for cplx_pat, val in new_initials.items():
                    if not isinstance(cplx_pat, (MonomerPattern,
                                                 ComplexPattern)):
                        raise SimulatorException('Dictionary key %s is not a '
                                                 'MonomerPattern or '
                                                 'ComplexPattern' %
                                                 repr(cplx_pat))
                self._initials = new_initials
            # accept vector of species amounts as an argument
            elif len(new_initials) != self._y[0].shape[1]:
                raise ValueError("y0 must be the same length as model.species")
            elif not isinstance(new_initials, np.ndarray):
                self._initials = np.array(new_initials)
            else:
                self._initials = new_initials

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
        y0 = np.zeros((len(self.model.species),))
        param_vals = self.param_values
        subs = dict((p, param_vals[i]) for i, p in
                    enumerate(self.model.parameters))

        def _set_initials(initials_source):
            for cp, value_obj in initials_source:
                cp = as_complex_pattern(cp)
                si = self.model.get_species_index(cp)
                if si is None:
                    raise IndexError("Species not found in model: %s" %
                                     repr(cp))
                # If this initial condition has already been set, skip it
                if y0[si] != 0:
                    continue
                if isinstance(value_obj, (int, float)):
                    value = value_obj
                elif value_obj in self.model.parameters:
                    pi = self.model.parameters.index(value_obj)
                    value = param_vals[pi]
                elif value_obj in self.model.expressions:
                    value = value_obj.expand_expr().evalf(subs=subs)
                else:
                    raise ValueError("Unexpected initial condition value type")
                y0[si] = value

        # Process any overrides
        if isinstance(self._initials, dict):
            _set_initials(self._initials.items())
        # Get remaining initials from the model itself
        _set_initials(self.model.initial_conditions)

        return y0

    @property
    def param_values(self):
        if self._params and not isinstance(self._params, dict):
            return self._params
        else:
            # create parameter vector from the values in the model
            param_values_dict = self._params if isinstance(self._params,
                                                           dict) else {}
            param_values = np.array([p.value for p in self.model.parameters])
            for key in param_values_dict.keys():
                try:
                    pi = self.model.parameters.index(self.model.parameters[
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
                if k not in self.model.parameters.keys():
                    raise IndexError("new_params dictionary has unknown "
                                     "parameter name (%s)" % k)
            self._params = new_params
        else:
            # accept vector of parameter values as an argument
            if len(new_params) != len(self.model.parameters):
                raise ValueError("new_params must be the same length as "
                                 "model.parameters")
            if isinstance(new_params, np.ndarray):
                self._params = new_params
            else:
                self._params = np.array(new_params)

    @classmethod
    def execute(cls, model, verbose=False, **kwargs):
        """
        Convenience method to instantiate a Simulator object, call its run()
        method, and return a ``yfull`` object of trajectories.

        Parameters
        ----------
        model : pysb.Model
            Model passed to the simulator constructor
        verbose : bool
            Print simulator messages to STDOUT
        **kwargs: dict
            Keyword arguments for the constructor
        """
        simulator = cls(model, verbose=verbose, **kwargs)
        simulator.run()
        return simulator.concs_all()

    @abstractmethod
    def run(self, tspan=None, param_values=None, initials=None):
        """Run a simulation.

        Returns nothing. Use appropriate methods to retrieve concentrations
        after simulation.
        """
        pass

    @property
    def n_sims(self):
        if self._y is None:
            return 0
        else:
            return len(self._y)

    def _calc_yobs_yexpr(self):
        """Calculate ``yobs`` and ``yexpr`` based on values of ``y`` obtained
        in ``run``."""
        nsims = self.n_sims
        exprs = self.model.expressions_dynamic()
        expr_names = [expr.name for expr in exprs]
        model_obs = self.model.observables
        obs_names = model_obs.keys()
        yobs_dtype = zip(obs_names, itertools.repeat(float)) if obs_names \
            else float
        yexpr_dtype = zip(expr_names, itertools.repeat(float)) if expr_names \
            else float

        self._yobs = [np.ndarray((len(self.tout[n]),),
                                 dtype=yobs_dtype) for n in range(nsims)]
        self._yobs_view = [self._yobs[n].view(float).
                           reshape(len(self._yobs[n]), -1) for n in range(
            nsims)]
        self._yexpr = [np.ndarray((len(self.tout[n]),),
                                  dtype=yexpr_dtype) for n in range(nsims)]
        self._yexpr_view = [self._yexpr[n].view(float).reshape(len(
            self._yexpr[n]), -1) for n in range(nsims)]
        param_values = self.param_values

        # loop over simulations
        for n in range(nsims):
            # observables
            for i, obs in enumerate(model_obs):
                self._yobs_view[n][:, i] = (
                    self._y[n][:, obs.species] * obs.coefficients).sum(axis=1)

            # expressions
            obs_dict = dict((k, self._yobs[n][k]) for k in obs_names)
            subs = dict((p, param_values[i]) for i, p in
                        enumerate(self.model.parameters))
            for i, expr in enumerate(exprs):
                expr_subs = expr.expand_expr().subs(subs)
                func = sympy.lambdify(obs_names, expr_subs, "numpy")
                self._yexpr_view[n][:, i] = func(**obs_dict)

    def concs_all(self, squeeze=True):
        """Aggregate species, observables, and expressions trajectories into
        a numpy.ndarray with record-style data-type for return to the user."""
        if self._yfull is None:
            if self._yobs is None:
                self._calc_yobs_yexpr()
            sp_names = ['__s%d' % i for i in range(len(self.model.species))]
            yfull_dtype = zip(sp_names, itertools.repeat(float))
            if len(self.model.observables):
                yfull_dtype += zip(self.model.observables.keys(),
                                   itertools.repeat(float))
            if len(self.model.expressions_dynamic()):
                yfull_dtype += zip(self.model.expressions_dynamic().keys(),
                                   itertools.repeat(float))
            yfull = len(self._y) * [None]
            # loop over simulations
            for n in range(self.n_sims):
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

        if self.n_sims == 1 and squeeze:
            return self._yfull[0]
        return self._yfull

    def concs_species(self, squeeze=True):
        if self.n_sims == 1 and squeeze:
            return self._y[0]
        else:
            return self._y

    def concs_observables(self, squeeze=True, tabular=False):
        if self._yobs is None:
            self._calc_yobs_yexpr()
        retval = self._yobs_view if tabular else self._yobs
        if self.n_sims == 1 and squeeze:
            return retval[0]
        else:
            return retval

    def concs_expressions(self, squeeze=True, tabular=False):
        if self._yexpr is None:
            self._calc_yobs_yexpr()
        retval = self._yexpr_view if tabular else self._yexpr
        if self.n_sims == 1 and squeeze:
            return retval[0]
        else:
            return retval
