import pysb.bng
import numpy
from scipy.integrate import ode
from scipy.weave import inline
import distutils.errors
import sympy
import re
import itertools


use_inline = False
# try to inline a C statement to see if inline is functional
try:
    inline('int i;', force=1)
    use_inline = True
except (distutils.errors.CompileError, ImportError):
    pass

# some sane default options for a few well-known integrators
default_integrator_options = {
    'vode': {
        'method': 'bdf',
        'with_jacobian': True,
        },
    'cvode': {
        'method': 'bdf',
        'iteration': 'newton',
        },
    }

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
    integrator : string, optional
        Name of the integrator to use, taken from the list of integrators known
        to :py:class:`scipy.integrate.ode`.
    integrator_params
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
        Observable trajectories. Length is is ``len(tspan)`` and record names
        follow ``model.observables`` names.
    yobs_view : numpy.ndarray
        An array view (sharing the same data buffer) on ``yobs``. Dimensionality
        is ``(len(tspan), len(model.observables))``.
    integrator : scipy.integrate.ode
        Integrator object.

    Notes
    -----
    The expensive step of generating the code for the right-hand side of the
    model's ODEs is performed during initialization. If you need to integrate
    the same model repeatedly with different parameters then you should build a
    single Solver object and then call its ``run`` method as needed.

    """

    def __init__(self, model, tspan, integrator='vode', **integrator_options):

        pysb.bng.generate_equations(model)

        code_eqs = '\n'.join(['ydot[%d] = %s;' % (i, sympy.ccode(model.odes[i])) for i in range(len(model.odes))])
        code_eqs = re.sub(r's(\d+)', lambda m: 'y[%s]' % (int(m.group(1))), code_eqs)
        for i, p in enumerate(model.parameters):
            code_eqs = re.sub(r'\b(%s)\b' % p.name, 'p[%d]' % i, code_eqs)

        # If we can't use weave.inline to run the C code, compile it as Python code instead for use with
        # exec. Note: C code with array indexing, basic math operations, and pow() just happens to also
        # be valid Python.  If the equations ever have more complex things in them, this might fail.
        if not use_inline:
            code_eqs_py = compile(code_eqs, '<%s odes>' % model.name, 'exec')

        def rhs(t, y, p):
            ydot = numpy.empty_like(y)
            # note that the evaluated code sets ydot as a side effect
            if use_inline:
                inline(code_eqs, ['ydot', 't', 'y', 'p']);
            else:
                exec code_eqs_py in locals()
            return ydot

        # build integrator options list from our defaults and any kwargs passed to this function
        options = {}
        try:
            options.update(default_integrator_options[integrator])
        except KeyError as e:
            pass
        options.update(integrator_options)

        self.model = model
        self.tspan = tspan
        self.y = numpy.ndarray((len(tspan), len(model.species)))
        if len(model.observables):
            self.yobs = numpy.ndarray(len(tspan), zip(model.observables.keys(),
                                                      itertools.repeat(float)))
        else:
            self.yobs = numpy.ndarray((len(tspan), 0))
        self.yobs_view = self.yobs.view(float).reshape(len(self.yobs), -1)
        self.integrator = ode(rhs).set_integrator(integrator, **options)


    def run(self, param_values=None, y0=None):
        """Perform an integration.

        Returns nothing; access the Solver object's ``y``, ``yobs``, or
        ``yobs_view`` attributes to retrieve the results.

        Parameters
        ----------
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

        if param_values is not None:
            # accept vector of parameter values as an argument
            if len(param_values) != len(self.model.parameters):
                raise Exception("param_values must be the same length as model.parameters")
            if not isinstance(param_values, numpy.ndarray):
                param_values = numpy.array(param_values)
        else:
            # create parameter vector from the values in the model
            param_values = numpy.array([p.value for p in self.model.parameters])

        if y0 is not None:
            # accept vector of species amounts as an argument
            if len(y0) != self.y.shape[1]:
                raise Exception("y0 must be the same length as model.species")
            if not isinstance(y0, numpy.ndarray):
                y0 = numpy.array(y0)
        else:
            y0 = numpy.zeros((self.y.shape[1],))
            for cp, ic_param in self.model.initial_conditions:
                pi = self.model.parameters.index(ic_param)
                si = self.model.get_species_index(cp)
                y0[si] = param_values[pi]

        # perform the actual integration
        self.integrator.set_initial_value(y0, self.tspan[0])
        self.integrator.set_f_params(param_values)
        self.y[0] = y0
        i = 1
        while self.integrator.successful() and self.integrator.t < self.tspan[-1]:
            self.y[i] = self.integrator.integrate(self.tspan[i])
            i += 1

        for i, obs in enumerate(self.model.observables):
            self.yobs_view[:, i] = \
                (self.y[:, obs.species] * obs.coefficients).sum(1)


def odesolve(model, tspan, param_values=None, y0=None, integrator='vode',
             **integrator_options):
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
    integrator_params
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
    >>> print yfull['A_total']   #doctest: +NORMALIZE_WHITESPACE
    [ 1.      0.899   0.8506  0.8179  0.793   0.7728  0.7557  0.7408  0.7277
    0.7158]

    Obtain a view on a returned record array which uses an atomic data-type and
    integer indexing (note that the view's data buffer is shared with the
    original array so there is no extra memory cost):

    >>> print yfull.shape
    (10,)
    >>> print yfull.dtype   #doctest: +NORMALIZE_WHITESPACE
    [('__s0', '<f8'), ('__s1', '<f8'), ('__s2', '<f8'), ('A_total', '<f8'),
    ('B_total', '<f8'), ('C_total', '<f8')]
    >>> print yfull[0:4, 1:3]
    Traceback (most recent call last):
      ...
    IndexError: too many indices
    >>> yarray = yfull.view(float).reshape(len(yfull), -1)
    >>> print yarray.shape
    (10, 6)
    >>> print yarray.dtype
    float64
    >>> print yarray[0:4, 1:3]
    [[  0.0000e+00   0.0000e+00]
     [  2.1672e-05   1.0093e-01]
     [  1.6980e-05   1.4943e-01]
     [  1.4502e-05   1.8209e-01]]

    """

    solver = Solver(model, tspan, integrator, **integrator_options)
    solver.run(param_values, y0)

    species_names = ['__s%d' % i for i in range(solver.y.shape[1])]
    yfull_dtype = zip(species_names, itertools.repeat(float))
    if len(solver.yobs.dtype):
        yfull_dtype += solver.yobs.dtype.descr
    yfull = numpy.ndarray(len(solver.y), yfull_dtype)

    yfull_view = yfull.view(float).reshape(len(yfull), -1)
    yfull_view[:, :solver.y.shape[1]] = solver.y
    yfull_view[:, solver.y.shape[1]:] = solver.yobs_view

    return yfull
