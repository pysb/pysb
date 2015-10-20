from abc import ABCMeta, abstractmethod
import numpy as np
import itertools

class Simulator:
    """An abstract base class for numerical simulation of models.

    Parameters
    ----------
    model : pysb.Model
        Model to simulate.
    tspan : vector-like, optional
        Time values over which to simulate. The first and last values define
        the time range. Returned trajectories are sampled at every value unless
        the simulation is interrupted for some reason, e.g., due to satisfaction 
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
    y : numpy.ndarray
        Species trajectories.
    yobs : numpy.ndarray with record-style data-type
        Record names follow ``model.observables`` names.
    yobs_view : numpy.ndarray
        An array view (sharing the same data buffer) on ``yobs``.
    yexpr : numpy.ndarray with record-style data-type
        Expression trajectories. Record names follow 
        ``model.expressions_dynamic()`` names.
    yexpr_view : numpy.ndarray
        An array view (sharing the same data buffer) on ``yexpr``.

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
    def __init__(self, model, tspan=None, verbose=False):
        self.model = model
        self.tspan = tspan
        self.verbose = verbose
        self.tout= None
        self.y = None
        self.yobs = None
        self.yobs_view = None
        self.yexpr = None
        self.yexpr_view = None
    
    @abstractmethod
    def run(self, tspan=None, param_values=None, y0=None):
        """Run a simulation.

        Returns nothing; access the Simulator object's ``y``, ``yobs``, 
        ``yobs_view``, ``yexpr``, or ``yexpr_view`` attributes to retrieve the 
        results.

        Parameters
        ----------
        tspan : vector-like, optional
            Time values over which to simulate. An Exception should be raised
            if not defined either here or in the constructor.
        param_values : vector-like, optional
            Values to use for every parameter in the model. Ordering is
            determined by the order of model.parameters. If not specified,
            parameter values will be taken directly from model.parameters.
        y0 : vector-like, optional
            Values to use for the initial condition of all species. Ordering is
            determined by the order of model.species. If not specified, initial
            conditions will be taken from model.initial_conditions (with initial
            condition parameter values taken from `param_values` if specified).

        """
        pass
    
    @abstractmethod
    def _calc_yobs_yexpr(self, param_values=None):
        """Calculate ``yobs`` and ``yexpr`` based on values of ``y`` obtained 
        in ``run``."""
        self.yobs = len(self.y) * [None]
        self.yobs_view = len(self.y) * [None]
        self.yexpr = len(self.y) * [None]
        self.yexpr_view = len(self.y) * [None]
        # loop over simulations
        for n in range(len(self.y)):
            # observables
            if len(self.model.observables):
                self.yobs[n] = np.ndarray(len(self.tout[n]), zip(self.model.observables.keys(), itertools.repeat(float)))
            else:
                self.yobs[n] = np.ndarray((len(self.tout[n]), 0))
            self.yobs_view[n] = self.yobs[n].view(float).reshape(len(self.yobs[n]), -1)
            for i,obs in enumerate(self.model.observables):
                self.yobs_view[n][:,i] = (self.y[n][:,obs.species] * obs.coefficients).sum(axis=1)
            # expressions
            exprs = self.model.expressions_dynamic()
            if len(exprs):
                self.yexpr[n] = np.ndarray(len(self.tout[n]), zip(exprs.keys(), itertools.repeat(float)))
            else:
                self.yexpr[n] = np.ndarray((len(self.tout[n]), 0))
            self.yexpr_view[n] = self.yexpr[n].view(float).reshape(len(self.yexpr[n]), -1)
            if param_values is None:
                param_values = [p.value for p in self.model.parameters]
            p_subs = { p.name : param_values[i] for i,p in enumerate(self.model.parameters) }
            obs_names = self.model.observables.keys()
            obs_dict = { name : self.yobs_view[n][:,i] for i,name in enumerate(obs_names) }
            for i,expr in enumerate(self.model.expressions_dynamic()):
                expr_subs = expr.expand_expr(self.model).subs(p_subs)
                func = sympy.lambdify(obs_names, expr_subs, "numpy")
                self.yexpr_view[n][:,i] = func(**obs_dict)
        self.yobs = np.array(self.yobs)
        self.yobs_view = np.array(self.yobs_view)
        self.yexpr = np.array(self.yexpr)
        self.yexpr_view = np.array(self.yexpr_view)
    
    @abstractmethod
    def get_yfull(self):
        """Aggregate species, observables, and expressions trajectories into
        a numpy.ndarray with record-style data-type for return to the user."""
        sp_names = ['__s%d' % i for i in range(len(self.model.species))] 
        yfull_dtype = zip(sp_names, itertools.repeat(float))
        if len(self.model.observables):
            yfull_dtype += zip(self.model.observables.keys(), itertools.repeat(float))
        if len(self.model.expressions_dynamic()):
            yfull_dtype += zip(self.model.expressions_dynamic().keys(), itertools.repeat(float))
        yfull = len(self.y) * [None]
        # loop over simulations
        for n in range(len(self.y)):
            yfull[n] = np.ndarray(len(self.tout[n]), yfull_dtype)
            yfull_view = yfull[n].view(float).reshape((len(yfull[n]), -1))
            n_sp = self.y[n].shape[1]
            n_ob = self.yobs_view[n].shape[1]
            n_ex = self.yexpr_view[n].shape[1]
            yfull_view[:,:n_sp] = self.y[n]
            yfull_view[:,n_sp:n_sp+n_ob] = self.yobs_view[n]
            yfull_view[:,n_sp+n_ob:n_sp+n_ob+n_ex] = self.yexpr_view[n]
        yfull = np.array(yfull)
        return yfull
