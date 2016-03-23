"""
A module containing a class that produces Python code for simulating a PySB
model without requiring PySB itself (note that NumPy and SciPy are still
required). This offers a way of distributing a model to those who do not have
PySB.

For information on how to use the model exporters, see the documentation
for :py:mod:`pysb.export`.

Structure of the standalone Python code
=======================================

The standalone Python code defines a class, ``Model``, with a method
``simulate`` that can be used to simulate the model.

As shown in the code for the Robertson model below, the ``Model`` class defines
the fields ``parameters``, ``observables``, and ``initial_conditions`` as lists
of ``collections.namedtuple`` objects that allow access to the features of the
model.

The ``simulate`` method has the following signature::

    def simulate(self, tspan, param_values=None, view=False):

with arguments as follows:

* ``tspan`` specifies the array of timepoints
* ``param_values`` is an optional vector of parameter values that can be used
  to override the nominal values defined in the PySB model
* ``view`` is an optional boolean argument that specifies if the simulation
  output arrays are returned as copies (views) of the original. If True,
  returns copies of the arrays, allowing changes to be made to values in the
  arrays without affecting the originals.

``simulate`` returns a tuple of two arrays. The first array is a matrix
with timecourses for each species in the model as the columns. The
second array is a numpy record array for the model's observables, which can
be indexed by name.

Output for the Robertson example model
======================================

Example code generated for the Robertson model, ``pysb.examples.robertson``:

.. literalinclude:: ../../examples/robertson_standalone.py

Using the standalone Python model
=================================

An example usage pattern for the standalone Robertson model, once generated::

    # Import the standalone model file
    import robertson_standalone
    import numpy
    from matplotlib import pyplot as plt

    # Instantiate the model object (the constructor takes no arguments)
    model = robertson_standalone.Model()

    # Simulate the model
    tspan = numpy.linspace(0, 100)
    (species_output, observables_output) = model.simulate(tspan)

    # Plot the results
    plt.figure()
    plt.plot(tspan, observables_output['A_total'])
    plt.show()
"""

import pysb
import pysb.bng
import sympy
import textwrap
from pysb.export import Exporter, pad
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
import re

class PythonExporter(Exporter):
    """A class for returning the standalone Python code for a given PySB model.

    Inherits from :py:class:`pysb.export.Exporter`, which implements
    basic functionality for all exporters.
    """
    def export(self):
        """Export Python code for simulation of a model without PySB.

        Returns
        -------
        string
            String containing the standalone Python code.
        """

        output = StringIO()
        pysb.bng.generate_equations(self.model)

        # Note: This has a lot of duplication from pysb.integrate.
        # Can that be helped?

        code_eqs = '\n'.join(['ydot[%d] = %s;' %
                                 (i, sympy.ccode(self.model.odes[i]))
                              for i in range(len(self.model.odes))])
        code_eqs = re.sub(r's(\d+)',
                          lambda m: 'y[%s]' % (int(m.group(1))), code_eqs)
        for i, p in enumerate(self.model.parameters):
            code_eqs = re.sub(r'\b(%s)\b' % p.name, 'p[%d]' % i, code_eqs)

        if self.docstring:
            output.write('"""')
            output.write(self.docstring)
            output.write('"""\n\n')
        output.write("# exported from PySB model '%s'\n" % self.model.name)
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
            'num_species': len(self.model.species),
            'num_params': len(self.model.parameters),
            'num_observables': len(self.model.observables),
            'num_ics': len(self.model.initial_conditions),
            }
        output.write(pad(r"""
            def __init__(self):
                self.y = None
                self.yobs = None
                self.integrator = scipy.integrate.ode(self.ode_rhs)
                self.integrator.set_integrator('vode', method='bdf',
                                               with_jacobian=True)
                self.y0 = numpy.empty(%(num_species)d)
                self.ydot = numpy.empty(%(num_species)d)
                self.sim_param_values = numpy.empty(%(num_params)d)
                self.parameters = [None] * %(num_params)d
                self.observables = [None] * %(num_observables)d
                self.initial_conditions = [None] * %(num_ics)d
            """, 4) % init_data)
        for i, p in enumerate(self.model.parameters):
            p_data = (i, repr(p.name), p.value)
            output.write(" " * 8)
            output.write("self.parameters[%d] = Parameter(%s, %.17g)\n" % p_data)
        output.write("\n")
        for i, obs in enumerate(self.model.observables):
            obs_data = (i, repr(obs.name), repr(obs.species),
                        repr(obs.coefficients))
            output.write(" " * 8)
            output.write("self.observables[%d] = Observable(%s, %s, %s)\n" %
                         obs_data)
        output.write("\n")
        for i, (cp, param) in enumerate(self.model.initial_conditions):
            ic_data = (i, self.model.parameters.index(param),
                       self.model.get_species_index(cp))
            output.write(" " * 8)
            output.write("self.initial_conditions[%d] = Initial(%d, %d)\n" %
                         ic_data)
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
                        raise Exception("param_values must have length %d" %
                                        len(self.parameters))
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
                        self.yobs = numpy.ndarray(len(tspan),
                                        zip((obs.name for obs in self.observables),
                                            itertools.repeat(float)))
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
            """, 4))

        return output.getvalue()


