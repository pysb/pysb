#!/usr/bin/env python
"Produce Python code for simulating a model without requiring pysb itself."


import pysb
import pysb.bng
import sympy
import re
import sys
import os
import textwrap
from StringIO import StringIO


def pad(text, depth=0):
    "Dedent multi-line string and pad with spaces."
    text = textwrap.dedent(text)
    text = re.sub(r'^(?m)', ' ' * depth, text)
    text += '\n'
    return text


def run(model, docstring=None):
    output = StringIO()
    pysb.bng.generate_equations(model)

    # Note: This has a lot of duplication from pysb.integrate. Can that be helped?

    code_eqs = '\n'.join(['ydot[%d] = %s;' % (i, sympy.ccode(model.odes[i])) for i in range(len(model.odes))])
    code_eqs = re.sub(r's(\d+)', lambda m: 'y[%s]' % (int(m.group(1))), code_eqs)
    for i, p in enumerate(model.parameters):
        code_eqs = re.sub(r'\b(%s)\b' % p.name, 'p[%d]' % i, code_eqs)

    output.write('"""')
    output.write(docstring)
    output.write('"""\n\n')
    output.write("# exported from PySB model '%s'\n" % model.name)
    output.write(pad(r"""
        import numpy
        import scipy.weave, scipy.integrate
        import collections
        import itertools
        import distutils.errors
        """))
    output.write(pad(r"""
        _use_inline = False
        # try to inline a C statement to see if inline is functional
        try:
            scipy.weave.inline('int i;', force=1)
            _use_inline = True
        except distutils.errors.CompileError:
            pass

        Parameter = collections.namedtuple('Parameter', 'name value')
        Observable = collections.namedtuple('Observable', 'name species coefficients')
        Initial = collections.namedtuple('Initial', 'param_index species_index')
        """))
    output.write("\n")

    output.write("class Model(object):\n")
    init_data = {
        'num_species': len(model.species),
        'num_params': len(model.parameters),
        'num_observables': len(model.observables),
        'num_ics': len(model.initial_conditions),
        }
    output.write(pad(r"""
        def __init__(self):
            self.y = None
            self.yobs = None
            self.integrator = scipy.integrate.ode(self.ode_rhs)
            self.integrator.set_integrator('vode', method='bdf', with_jacobian=True)
            self.y0 = numpy.empty(%(num_species)d)
            self.ydot = numpy.empty(%(num_species)d)
            self.sim_param_values = numpy.empty(%(num_params)d)
            self.parameters = [None] * %(num_params)d
            self.observables = [None] * %(num_observables)d
            self.initial_conditions = [None] * %(num_ics)d
        """, 4) % init_data)
    for i, p in enumerate(model.parameters):
        p_data = (i, repr(p.name), p.value)
        output.write(" " * 8)
        output.write("self.parameters[%d] = Parameter(%s, %g)\n" % p_data)
    output.write("\n")
    for i, obs in enumerate(model.observables):
        obs_data = (i, repr(obs.name), repr(obs.species), repr(obs.coefficients))
        output.write(" " * 8)
        output.write("self.observables[%d] = Observable(%s, %s, %s)\n" % obs_data)
    output.write("\n")
    for i, (cp, param) in enumerate(model.initial_conditions):
        ic_data = (i, model.parameters.index(param), model.get_species_index(cp))
        output.write(" " * 8)
        output.write("self.initial_conditions[%d] = Initial(%d, %d)\n" % ic_data)
    output.write("\n")

    output.write("    if _use_inline:\n")
    output.write(pad(r"""
        def ode_rhs(self, t, y, p):
            ydot = self.ydot
            scipy.weave.inline(r'''%s''', ['ydot', 't', 'y', 'p'])
            return ydot
        """, 8) % (pad('\n' + code_eqs, 16) + ' ' * 16))
    output.write("    else:\n")
    output.write(pad(r"""
        def ode_rhs(self, t, y, p):
            ydot = self.ydot
            %s
            return ydot
        """, 8) % pad('\n' + code_eqs, 12).replace(';','').strip())

    # note the simulate method is fixed, i.e. it doesn't require any templating
    output.write(pad(r"""
        def simulate(self, tspan, param_values=None, view=False):
            if param_values is not None:
                # accept vector of parameter values as an argument
                if len(param_values) != len(self.parameters):
                    raise Exception("param_values must have length %d" % len(self.parameters))
                self.sim_param_values[:] = param_values
            else:
                # create parameter vector from the values in the model
                self.sim_param_values[:] = [p.value for p in self.parameters]
            self.y0.fill(0)
            for ic in self.initial_conditions:
                self.y0[ic.species_index] = self.sim_param_values[ic.param_index]
            if self.y is None or len(tspan) != len(self.y):
                self.y = numpy.empty((len(tspan), len(self.y0)))
                if len(self.observables):
                    self.yobs = numpy.ndarray(len(tspan), zip((obs.name for obs in self.observables),
                                                              itertools.repeat(float)))
                else:
                    self.yobs = numpy.ndarray((len(tspan), 0))
                self.yobs_view = self.yobs.view(float).reshape(len(self.yobs), -1)
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
        """, 4))

    return output.getvalue()


if __name__ == '__main__':
    # sanity checks on filename
    if len(sys.argv) <= 1:
        raise Exception("You must specify the filename of a model script")
    model_filename = sys.argv[1]
    if not os.path.exists(model_filename):
        raise Exception("File '%s' doesn't exist" % model_filename)
    if not re.search(r'\.py$', model_filename):
        raise Exception("File '%s' is not a .py file" % model_filename)
    sys.path.insert(0, os.path.dirname(model_filename))
    model_name = re.sub(r'\.py$', '', os.path.basename(model_filename))
    # import it
    try:
        # FIXME if the model has the same name as some other "real" module
        # which we use, there will be trouble
        # (use the imp package and import as some safe name?)
        model_module = __import__(model_name)
    except StandardError as e:
        print "Error in model script:\n"
        raise
    # grab the 'model' variable from the module
    try:
        model = model_module.__dict__['model']
    except KeyError:
        raise Exception("File '%s' isn't a model file" % model_filename)
    print run(model, model_module.__doc__)
