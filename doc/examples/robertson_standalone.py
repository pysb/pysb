"""A simple three-species chemical kinetics system known as "Robertson's
example", as presented in:

H. H. Robertson, The solution of a set of reaction rate equations, in Numerical
Analysis: An Introduction, J. Walsh, ed., Academic Press, 1966, pp. 178-182.
"""

# exported from PySB model 'robertson'

import numpy
import scipy.integrate
import collections
import itertools
import distutils.errors


_use_cython = False
# try to inline a C statement to see if Cython is functional
try:
    import Cython
except ImportError:
    Cython = None
if Cython:
    from Cython.Compiler.Errors import CompileError
    try:
        Cython.inline('x = 1', force=True, quiet=True)
        _use_cython = True
    except (CompileError,
            distutils.errors.CompileError,
            ValueError):
        pass

Parameter = collections.namedtuple('Parameter', 'name value')
Observable = collections.namedtuple('Observable', 'name species coefficients')
Initial = collections.namedtuple('Initial', 'param_index species_index')


class Model(object):
    def __init__(self):
        self.y = None
        self.yobs = None
        self.y0 = numpy.empty(3)
        self.ydot = numpy.empty(3)
        self.sim_param_values = numpy.empty(6)
        self.parameters = [None] * 6
        self.observables = [None] * 3
        self.initials = [None] * 3

        self.parameters[0] = Parameter('k1', 0.040000000000000001)
        self.parameters[1] = Parameter('k2', 30000000)
        self.parameters[2] = Parameter('k3', 10000)
        self.parameters[3] = Parameter('A_0', 1)
        self.parameters[4] = Parameter('B_0', 0)
        self.parameters[5] = Parameter('C_0', 0)

        self.observables[0] = Observable('A_total', [0], [1])
        self.observables[1] = Observable('B_total', [1], [1])
        self.observables[2] = Observable('C_total', [2], [1])

        self.initials[0] = Initial(3, 0)
        self.initials[1] = Initial(4, 1)
        self.initials[2] = Initial(5, 2)

        code_eqs = '''
ydot[0] = (y[0]*p[0])*(-1.0) + (y[1]*y[2]*p[2])*1.0
ydot[1] = (y[0]*p[0])*1.0 + (pow(y[1], 2)*p[1])*(-1.0) + (y[1]*y[2]*p[2])*(-1.0)
ydot[2] = (pow(y[1], 2)*p[1])*1.0
'''
        if _use_cython:

            def ode_rhs(t, y, p):
                ydot = self.ydot
                Cython.inline(code_eqs, quiet=True)
                return ydot

        else:

            def ode_rhs(t, y, p):
                ydot = self.ydot
                exec(code_eqs)
                return ydot

        self.integrator = scipy.integrate.ode(ode_rhs)
        self.integrator.set_integrator('vode', method='bdf', with_jacobian=True)

    def simulate(self, tspan, param_values=None, view=False):
        if param_values is not None:
            # accept vector of parameter values as an argument
            if len(param_values) != len(self.parameters):
                raise Exception("param_values must have length %d" %
                                len(self.parameters))
            self.sim_param_values[:] = param_values
        else:
            # create parameter vector from the values in the model
            self.sim_param_values[:] = [p.value for p in self.parameters]
        self.y0.fill(0)
        for ic in self.initials:
            self.y0[ic.species_index] = self.sim_param_values[ic.param_index]
        if self.y is None or len(tspan) != len(self.y):
            self.y = numpy.empty((len(tspan), len(self.y0)))
            if len(self.observables):
                self.yobs = numpy.ndarray(len(tspan),
                                list(zip((obs.name for obs in self.observables),
                                    itertools.repeat(float))))
            else:
                self.yobs = numpy.ndarray((len(tspan), 0))
            self.yobs_view = self.yobs.view(float).reshape(len(self.yobs),
                                                           -1)
        # perform the actual integration
        self.integrator.set_initial_value(self.y0, tspan[0])
        self.integrator.set_f_params(self.sim_param_values)
        self.y[0] = self.y0
        t = 1
        while self.integrator.successful() and self.integrator.t < tspan[-1]:
            self.y[t] = self.integrator.integrate(tspan[t])
            t += 1
        for i, obs in enumerate(self.observables):
            self.yobs_view[:, i] = \
                (self.y[:, obs.species] * obs.coefficients).sum(1)
        if view:
            y_out = self.y.view()
            yobs_out = self.yobs.view()
            for a in y_out, yobs_out:
                a.flags.writeable = False
        else:
            y_out = self.y.copy()
            yobs_out = self.yobs.copy()
        return (y_out, yobs_out)
    

