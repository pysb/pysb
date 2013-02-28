#!/usr/bin/env python
"""
A module for converting a PySB model to an equivalent set of ordinary
differential equations for integration or analysis in PottersWheel_.  Can be
used as a command-line script or from within the Python shell.

.. _PottersWheel: http://www.potterswheel.de

Usage as a command-line script
==============================

As a command-line script, run as follows::

    export_potterswheel.py model_name.py > model_name.m

where ``model_name.py`` contains a PySB model definition (i.e., contains an
instance of ``pysb.core.Model`` instantiated as a global variable). The text of
the PottersWheel code will be printed to standard out, allowing it to be
redirected to another file, as shown in this example. Note that the name of the
``.m`` file must match the name of the ODE function (e.g., ``robertson.m`` in
the example below).

Usage in the Python shell
=========================

To use in a Python shell, import a model::

    from pysb.examples.robertson import model

and import this module::

    from pysb.tools import export_potterswheel

then call the function ``run``, passing the model instance::

    potterswheel_output = export_potterswheel.run(model)

then write the output to a file::

    f = open('robertson.m', 'w')
    f.write(potterswheel_output)
    f.close()

Output for the Robertson example model
======================================

The PottersWheel code produced will follow the form as given below for
``pysb.examples.robertson``::

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
from StringIO import StringIO


def run(model):
    """Export the model as a set of ODEs for use in PottersWheel.

    Parameters
    ----------
    model : pysb.core.Model
        The model to export as a set of ODEs.

    Returns
    -------
    string
        String containing the PottersWheel code for the ODEs.
    """

    output = StringIO()
    pysb.bng.generate_equations(model)

    ic_values = [0] * len(model.odes)
    for cp, ic_param in model.initial_conditions:
        ic_values[model.get_species_index(cp)] = ic_param.value

    # list of "dynamic variables"
    pw_x = ["m = pwAddX(m, 's%d', %e);" % (i, ic_values[i])
            for i in range(len(model.odes))]

    # parameters
    pw_k = ["m = pwAddK(m, '%s', %e);" % (p.name, p.value)
            for p in model.parameters]

    # equations (one for each dynamic variable)
    # Note that we just generate C code, which for basic math expressions
    # is identical to matlab.  We just have to change 'pow' to 'power'.
    # Ideally there would be a matlab formatter for sympy.
    pw_ode = ["m = pwAddODE(m, 's%d', '%s');" % (i, sympy.ccode(model.odes[i])) for i in range(len(model.odes))]
    pw_ode = [re.sub(r'pow(?=\()', 'power', s) for s in pw_ode]

    # observables
    pw_y = ["m = pwAddY(m, '%s', '%s');" % (obs.name, ' + '.join('%f * s%s' % t for t in zip(obs.coefficients, obs.species))) for obs in model.observables]

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
    print run(model)
