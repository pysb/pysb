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
except distutils.errors.CompileError as e:
    pass

# try to load our pysundials cvode wrapper for scipy.integrate.ode
try:
    import pysb.pysundials_helpers
except ImportError as e:
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
        self.yobs = numpy.ndarray(len(tspan), zip(model.observables.keys(),
                                                  itertools.repeat(float)))
        self.yobs_view = self.yobs.view(float).reshape(len(self.yobs), -1)
        self.integrator = ode(rhs).set_integrator(integrator, **options)


    def run(self, param_values=None, y0=None):

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


def odesolve(model, t, param_values=None, y0=None, integrator='vode',
             **integrator_options):
    solver = Solver(model, t, integrator, **integrator_options)
    solver.run(param_values, y0)

    species_names = ['__s%d' % i for i in range(solver.y.shape[1])]
    species_dtype = zip(species_names, itertools.repeat(float))
    yfull_dtype = species_dtype + solver.yobs.dtype.descr
    yfull = numpy.ndarray(len(solver.y), yfull_dtype)

    yfull_view = yfull.view(float).reshape(len(yfull), -1)
    yfull_view[:, :solver.y.shape[1]] = solver.y
    yfull_view[:, solver.y.shape[1]:] = solver.yobs_view

    return yfull
