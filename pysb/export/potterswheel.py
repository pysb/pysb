"""
Module containing a class for converting a PySB model to an equivalent set of
ordinary differential equations for integration or analysis in PottersWheel_.

.. _PottersWheel: http://www.potterswheel.de

For information on how to use the model exporters, see the documentation
for :py:mod:`pysb.export`.

Output for the Robertson example model
======================================

The PottersWheel code produced will follow the form as given below for
``pysb.examples.robertson``::

    % A simple three-species chemical kinetics system known as "Robertson's
    % example", as presented in:
    % 
    % H. H. Robertson, The solution of a set of reaction rate equations, in Numerical
    % Analysis: An Introduction, J. Walsh, ed., Academic Press, 1966, pp. 178-182.
    % 
    % PottersWheel model definition file
    % save as robertson.m
    function m = robertson()

    m = pwGetEmptyModel();

    % meta information
    m.ID          = 'robertson';
    m.name        = 'robertson';
    m.description = '';
    m.authors     = {''};
    m.dates       = {''};
    m.type        = 'PW-1-5';

    % dynamic variables
    m = pwAddX(m, 's0', 1.000000e+00);
    m = pwAddX(m, 's1', 0.000000e+00);
    m = pwAddX(m, 's2', 0.000000e+00);

    % dynamic parameters
    m = pwAddK(m, 'k1', 4.000000e-02);
    m = pwAddK(m, 'k2', 3.000000e+07);
    m = pwAddK(m, 'k3', 1.000000e+04);
    m = pwAddK(m, 'A_0', 1.000000e+00);
    m = pwAddK(m, 'B_0', 0.000000e+00);
    m = pwAddK(m, 'C_0', 0.000000e+00);

    % ODEs
    m = pwAddODE(m, 's0', '-k1*s0 + k3*s1*s2');
    m = pwAddODE(m, 's1', 'k1*s0 - k2*power(s1, 2) - k3*s1*s2');
    m = pwAddODE(m, 's2', 'k2*power(s1, 2)');

    % observables
    m = pwAddY(m, 'A_total', '1.000000 * s0');
    m = pwAddY(m, 'B_total', '1.000000 * s1');
    m = pwAddY(m, 'C_total', '1.000000 * s2');

    % end of PottersWheel model robertson
"""

import pysb
import pysb.bng
import sympy
import re
import sys
import os
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
from pysb.export import Exporter

class PottersWheelExporter(Exporter):
    """A class for returning the PottersWheel equivalent for a given PySB model.

    Inherits from :py:class:`pysb.export.Exporter`, which implements
    basic functionality for all exporters.
    """

    def export(self):
        """Generate the PottersWheel code for the ODEs of the PySB model
        associated with the exporter.

        Returns
        -------
        string
            String containing the PottersWheel code for the ODEs.
        """

        output = StringIO()
        pysb.bng.generate_equations(self.model)

        model_name = self.model.name.replace('.', '_')

        ic_values = [0] * len(self.model.odes)
        for cp, ic_param in self.model.initial_conditions:
            ic_values[self.model.get_species_index(cp)] = ic_param.value

        # list of "dynamic variables"
        pw_x = ["m = pwAddX(m, 's%d', %e);" % (i, ic_values[i])
                for i in range(len(self.model.odes))]

        # parameters
        pw_k = ["m = pwAddK(m, '%s', %e);" % (p.name, p.value)
                for p in self.model.parameters]

        # equations (one for each dynamic variable)
        # Note that we just generate C code, which for basic math expressions
        # is identical to matlab.  We just have to change 'pow' to 'power'.
        # Ideally there would be a matlab formatter for sympy.
        pw_ode = ["m = pwAddODE(m, 's%d', '%s');" %
                  (i, sympy.ccode(self.model.odes[i]))
                  for i in range(len(self.model.odes))]
        pw_ode = [re.sub(r'pow(?=\()', 'power', s) for s in pw_ode]

        # observables
        pw_y = ["m = pwAddY(m, '%s', '%s');" %
                (obs.name,
                    ' + '.join('%f * s%s' % t
                               for t in zip(obs.coefficients, obs.species)))
                 for obs in self.model.observables]

        # Add docstring, if present
        if self.docstring:
            output.write('% ' + self.docstring.replace('\n', '\n% '))
            output.write('\n')

        output.write('% PottersWheel model definition file\n')
        output.write('%% save as %s.m\n' % model_name)
        output.write('function m = %s()\n' % model_name)
        output.write('\n')
        output.write('m = pwGetEmptyModel();\n')
        output.write('\n')
        output.write('% meta information\n')
        output.write("m.ID          = '%s';\n" % model_name)
        output.write("m.name        = '%s';\n" % model_name)
        output.write("m.description = '';\n")
        output.write("m.authors     = {''};\n")
        output.write("m.dates       = {''};\n")
        output.write("m.type        = 'PW-1-5';\n")
        output.write('\n')
        output.write('% dynamic variables\n')
        for x in pw_x:
            output.write(x)
            output.write('\n')
        output.write('\n')
        output.write('% dynamic parameters\n')
        for k in pw_k:
            output.write(k)
            output.write('\n')
        output.write('\n')
        output.write('% ODEs\n')
        for ode in pw_ode:
            output.write(ode)
            output.write('\n')
        output.write('\n')
        output.write('% observables\n')
        for y in pw_y:
            output.write(y)
            output.write('\n')
        output.write('\n')
        output.write('%% end of PottersWheel model %s\n' % model_name)
        return output.getvalue()

