import numpy
from pysb.simulator import ScipyOdeSimulator


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
        An array view (sharing the same data buffer) on ``yobs``.
        Dimensionality is ``(len(tspan), len(model.observables))``.
    yexpr : numpy.ndarray with record-style data-type
        Expression trajectories. Length is ``len(tspan)`` and record names
        follow ``model.expressions_dynamic()`` names.
    yexpr_view : numpy.ndarray
        An array view (sharing the same data buffer) on ``yexpr``.
        Dimensionality is ``(len(tspan), len(model.expressions_dynamic()))``.
    integrator : scipy.integrate.ode
        Integrator object.

    Notes
    -----
    The expensive step of generating the code for the right-hand side of the
    model's ODEs is performed during initialization. If you need to integrate
    the same model repeatedly with different parameters then you should build a
    single Solver object and then call its ``run`` method as needed.

    """
    def __init__(self, model, tspan, use_analytic_jacobian=False,
                 integrator='vode', cleanup=True,
                 verbose=False, **integrator_options):
        self._sim = ScipyOdeSimulator(model, verbose=verbose, tspan=tspan,
                                     use_analytic_jacobian=
                                     use_analytic_jacobian,
                                     integrator=integrator, cleanup=cleanup,
                                     **integrator_options)
        self.result = None
        self._yexpr_view = None
        self._yobs_view = None

    @property
    def _use_inline(self):
        return ScipyOdeSimulator._use_inline

    @_use_inline.setter
    def _use_inline(self, use_inline):
        ScipyOdeSimulator._use_inline = use_inline

    @property
    def y(self):
        return self.result.species if self.result is not None else None

    @property
    def yobs(self):
        return self.result.observables if self.result is not None else None

    @property
    def yobs_view(self):
        if self._yobs_view is None:
            self._yobs_view = self.yobs.view(float).reshape(len(self.yobs), -1)
        return self._yobs_view

    @property
    def yexpr(self):
        return self.result.expressions if self.result is not None else None

    @property
    def yexpr_view(self):
        if self._yexpr_view is None:
            self._yexpr_view = self.yexpr.view(float).reshape(len(self.yexpr),
                                                              -1)
        return self._yexpr_view

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
            conditions will be taken from model.initial_conditions (with
            initial condition parameter values taken from `param_values` if
            specified).
        """
        self._yobs_view = None
        self._yexpr_view = None
        self.result = self._sim.run(param_values=param_values, initials=y0)


def odesolve(model, tspan, param_values=None, y0=None, integrator='vode',
             cleanup=True, verbose=False, **integrator_options):
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
    cleanup : bool, optional
        Remove temporary files after completion if True. Set to False for
        debugging purposes.
    verbose : bool, optionsal
        Increase verbosity of simulator output.
    integrator_options :
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
    [1.      0.899   0.8506  0.8179  0.793   0.7728  0.7557  0.7408  0.7277
    0.7158]

    Obtain a view on a returned record array which uses an atomic data-type and
    integer indexing (note that the view's data buffer is shared with the
    original array so there is no extra memory cost):

    >>> yfull.shape == (10, )
    True
    >>> print(yfull.dtype)                 #doctest: +NORMALIZE_WHITESPACE
    [('__s0', '<f8'), ('__s1', '<f8'), ('__s2', '<f8'), ('A_total', '<f8'),
    ('B_total', '<f8'), ('C_total', '<f8')]
    >>> print(yfull[0:4, 1:3])             #doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    IndexError: too many indices...
    >>> yarray = yfull.view(float).reshape(len(yfull), -1)
    >>> yarray.shape == (10, 6)
    True
    >>> print(yarray.dtype)
    float64
    >>> print(yarray[0:4, 1:3])            #doctest: +NORMALIZE_WHITESPACE
    [[0.0000e+00   0.0000e+00]
     [2.1672e-05   1.0093e-01]
     [1.6980e-05   1.4943e-01]
     [1.4502e-05   1.8209e-01]]

    """
    integrator_options['integrator'] = integrator
    sim = ScipyOdeSimulator(model, tspan=tspan, cleanup=cleanup,
                            verbose=verbose, **integrator_options)
    simres = sim.run(param_values=param_values, initials=y0)
    return simres.all
