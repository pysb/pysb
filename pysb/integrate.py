import pysb.core
import pysb.bng
import numpy
from scipy.integrate import ode
from scipy.weave import inline
import scipy.weave.build_tools
import distutils.errors
import sympy
import re
import itertools

# some sane default options for a few well-known integrators
default_integrator_options = {
    'vode': {
        'method': 'bdf',
        'with_jacobian': True,
        # Set nsteps as high as possible to give our users flexibility in
        # choosing their time step. (Let's be safe and assume vode was compiled
        # with 32-bit ints. What would actually happen if it was and we passed
        # 2**64-1 though?)
        'nsteps': 2**31 - 1,
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
        An array view (sharing the same data buffer) on ``yobs``. Dimensionality
        is ``(len(tspan), len(model.observables))``.
    yexpr : numpy.ndarray with record-style data-type
        Expression trajectories. Length is ``len(tspan)`` and record names
        follow ``model.expressions_dynamic()`` names.
    yexpr_view : numpy.ndarray
        An array view (sharing the same data buffer) on ``yexpr``. Dimensionality
        is ``(len(tspan), len(model.expressions_dynamic()))``.
    integrator : scipy.integrate.ode
        Integrator object.

    Notes
    -----
    The expensive step of generating the code for the right-hand side of the
    model's ODEs is performed during initialization. If you need to integrate
    the same model repeatedly with different parameters then you should build a
    single Solver object and then call its ``run`` method as needed.

    """

    @staticmethod
    def _test_inline():
        """Detect whether scipy.weave.inline is functional."""
        if not hasattr(Solver, '_use_inline'):
            Solver._use_inline = False
            try:
                inline('int i;', force=1)
                Solver._use_inline = True
            except (scipy.weave.build_tools.CompileError,
                    distutils.errors.CompileError, ImportError):
                pass

    def __init__(self, model, tspan, integrator='vode', cleanup=True, verbose=False, **integrator_options):
        
        self.verbose = verbose
        pysb.bng.generate_equations(model,cleanup,self.verbose)
        
        code_eqs = '\n'.join(['ydot[%d] = %s;' % (i, sympy.ccode(model.odes[i])) for i in range(len(model.odes))])
        
        for e in model.expressions:
            code_eqs = re.sub(r'\b(%s)\b' % e.name, '('+sympy.ccode(e.expand_expr())+')', code_eqs)

        for obs in model.observables:
            obs_string = ''
            for i in range(len(obs.coefficients)):
                if i > 0: obs_string += "+"
                if obs.coefficients[i] > 1: obs_string += str(obs.coefficients[i])+"*"
                obs_string += "__s"+str(obs.species[i])
            if len(obs.coefficients) > 1: obs_string = '(' + obs_string + ')'
            code_eqs = re.sub(r'\b(%s)\b' % obs.name, obs_string, code_eqs)

        code_eqs = re.sub(r'_*s(\d+)', lambda m: 'y[%s]' % (int(m.group(1))), code_eqs)

        for i, p in enumerate(model.parameters):
            code_eqs = re.sub(r'\b(%s)\b' % p.name, 'p[%d]' % i, code_eqs)
        
        Solver._test_inline()
        
        # If we can't use weave.inline to run the C code, compile it as Python code instead for use with
        # exec. Note: C code with array indexing, basic math operations, and pow() just happens to also
        # be valid Python.  If the equations ever have more complex things in them, this might fail.
        if not Solver._use_inline:
            code_eqs_py = compile(code_eqs, '<%s odes>' % model.name, 'exec')
        else:
            for arr_name in ('ydot', 'y', 'p'):
                macro = arr_name.upper() + '1'
                code_eqs = re.sub(r'\b%s\[(\d+)\]' % arr_name,'%s(\\1)' % macro, code_eqs)
        
        def rhs(t, y, p):
            ydot = self.ydot
            # note that the evaluated code sets ydot as a side effect
            if Solver._use_inline:
                inline(code_eqs, ['ydot', 't', 'y', 'p']);
            else:
                exec code_eqs_py in locals()
            return ydot

        # build integrator options list from our defaults and any kwargs passed to this function
        options = {}
        if default_integrator_options.get(integrator):
            options.update(default_integrator_options[integrator]) # default options
#         try:
#             options.update(default_integrator_options[integrator])
#         except KeyError as e:
#             pass
        
        options.update(integrator_options) # overwrite defaults
        self.model = model
        self.tspan = tspan
        self.y = numpy.ndarray((len(self.tspan), len(self.model.species))) # species concentrations
        self.ydot = numpy.ndarray(len(self.model.species))
        
        # observables
        if len(self.model.observables):
            self.yobs = numpy.ndarray(len(self.tspan), zip(self.model.observables.keys(),
                                                      itertools.repeat(float)))
        else:
            self.yobs = numpy.ndarray((len(self.tspan), 0))
        self.yobs_view = self.yobs.view(float).reshape(len(self.yobs), -1)
        
        # expressions
        exprs = self.model.expressions_dynamic()
        if len(exprs):
            self.yexpr = numpy.ndarray(len(self.tspan), zip(exprs.keys(),
                                                       itertools.repeat(float)))
        else:
            self.yexpr = numpy.ndarray((len(self.tspan), 0))
        self.yexpr_view = self.yexpr.view(float).reshape(len(self.yexpr), -1)
        
        # Integrator
        if scipy.integrate._ode.find_integrator(integrator):
            self.integrator = ode(rhs).set_integrator(integrator, **options)
        else:
            raise Exception("Integrator type '" + integrator + "' not recognized.")
        
#         help(self.integrator)
#         for (key, val) in self.integrator.__dict__['_integrator'].__dict__.items():
#             print key, ": ", val
#         quit()

    def run(self, param_values=None, y0=None,initial_changes=None):
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

#         subs = dict((p, param_values[i]) for i, p in enumerate(self.model.parameters))
        subs = dict((p.name, param_values[i]) for i, p in enumerate(self.model.parameters))
        
        if y0 is not None:
            # accept vector of species amounts as an argument
            if len(y0) != self.y.shape[1]:
                raise Exception("y0 must be the same length as model.species")
            if not isinstance(y0, numpy.ndarray):
                y0 = numpy.array(y0)
        elif initial_changes is not None:
            y0 = numpy.zeros((self.y.shape[1],))
            for cp, value_obj in self.model.initial_conditions:
                if value_obj.name in initial_changes:
                    value = initial_changes[value_obj.name]
                elif value_obj in self.model.parameters:
                    pi = self.model.parameters.index(value_obj)
                    value = param_values[pi]
                elif value_obj in self.model.expressions:
                    value = value_obj.expand_expr(self.model).evalf(subs=subs)
                else:
                    raise ValueError("Unexpected initial condition value type")
                si = self.model.get_species_index(cp)
                if si is None:
                    raise Exception("Species not found in model: %s" % repr(cp))
                y0[si] = value
        else:
            y0 = numpy.zeros((self.y.shape[1],))
            for cp, value_obj in self.model.initial_conditions:
                if value_obj in self.model.parameters:
                    pi = self.model.parameters.index(value_obj)
                    value = param_values[pi]
                elif value_obj in self.model.expressions:
                    value = value_obj.expand_expr(self.model).evalf(subs=subs)
                else:
                    raise ValueError("Unexpected initial condition value type")
                si = self.model.get_species_index(cp)
                if si is None:
                    raise Exception("Species not found in model: %s" % repr(cp))
                y0[si] = value

        # perform the actual integration
        self.integrator.set_initial_value(y0, self.tspan[0])
        self.integrator.set_f_params(param_values)
        self.y[0] = y0
        i = 1
        if self.verbose: 
            print "Integrating..."
            print "\t"+"Time"
            print "\t"+"----"
            print "\t", self.integrator.t
        while self.integrator.successful() and self.integrator.t < self.tspan[-1]:
            self.y[i] = self.integrator.integrate(self.tspan[i]) # integration
            i += 1
            ######
#             self.integrator.integrate(self.tspan[i],step=True)
#             if self.integrator.t >= self.tspan[i]: i += 1
            ######
            if self.verbose: print "\t", self.integrator.t
        if self.verbose: print "...Done."
        if self.integrator.t < self.tspan[-1]:
            self.y[i:, :] = 'nan'

        # calculate observables
        for i, obs in enumerate(self.model.observables): 
            self.yobs_view[:, i] = (self.y[:, obs.species] * obs.coefficients).sum(1)
        
        # calculate expressions
        obs_names = self.model.observables.keys()
        obs_dict = dict((k, self.yobs[k]) for k in obs_names)
        for expr in self.model.expressions_dynamic():
            expr_subs = expr.expand_expr().subs(subs)
            func = sympy.lambdify(obs_names, expr_subs, "numpy")
            self.yexpr[expr.name] = func(**obs_dict)

def odesolve(model, tspan, param_values=None, y0=None, integrator='vode', cleanup=True, verbose=False,
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
    >>> print yfull[0:4, 1:3]   #doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    IndexError: too many indices...
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

    solver = Solver(model, tspan, integrator, cleanup, verbose, **integrator_options)
    solver.run(param_values, y0)

    species_names = ['__s%d' % i for i in range(solver.y.shape[1])]
    yfull_dtype = zip(species_names, itertools.repeat(float))
    if len(solver.yobs.dtype):
        yfull_dtype += solver.yobs.dtype.descr
    if len(solver.yexpr.dtype):
        yfull_dtype += solver.yexpr.dtype.descr
    yfull = numpy.ndarray(len(solver.y), yfull_dtype)

    n_sp = solver.y.shape[1]
    n_ob = solver.yobs_view.shape[1]
    n_ex = solver.yexpr_view.shape[1]

    yfull_view = yfull.view(float).reshape(len(yfull), -1)
    yfull_view[:,:n_sp] = solver.y
    yfull_view[:,n_sp:n_sp+n_ob] = solver.yobs_view
    yfull_view[:,n_sp+n_ob:n_sp+n_ob+n_ex] = solver.yexpr_view

    return yfull


def setup_module(module):
    """Doctest fixture for nose."""
    # Distutils' temp directory creation code has a more-or-less unsuppressable
    # print to stdout which will break the doctest which triggers it (via
    # scipy.weave.inline). So here we run an end-to-end test of the inlining
    # system to get that print out of the way at a point where it won't matter.
    # As a bonus, the test harness is suppressing writes to stdout at this time
    # anyway so the message is just swallowed silently.
    Solver._test_inline()
