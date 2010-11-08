#!/usr/bin/env python

import pysb
import pysb.bng
import sympy
import re
import sys
import os
from StringIO import StringIO


def run(model):
    output = StringIO()
    pysb.bng.generate_equations(model)

    obs_names = [name for name, rp in model.observable_patterns]

    ic_values = [0] * len(model.odes)
    for cp, ic_param in model.initial_conditions:
        ic_values[model.get_species_index(cp)] = ic_param.value

    # list of "dynamic variables"
    pw_x = ["m = pwAddX(m, 's%d', %e);" % (i, ic_values[i]) for i in range(len(model.odes))]

    # parameters
    pw_k = ["m = pwAddK(m, '%s', %e);" % (p.name, p.value) for p in model.parameters]

    # equations (one for each dynamic variable)
    # Note that we just generate C code, which for basic math expressions
    # is identical to matlab.  We just have to change 'pow' to 'power'.
    # Ideally there would be a matlab formatter for sympy.
    pw_ode = ["m = pwAddODE(m, 's%d', '%s');" % (i, sympy.ccode(model.odes[i])) for i in range(len(model.odes))]
    pw_ode = [re.sub(r'pow(?=\()', 'power', s) for s in pw_ode]

    # observables
    pw_y = ["m = pwAddY(m, '%s', '%s');" % (' + '.join(['%f * s%s' % g for g in model.observable_groups[name]]), name) for name in obs_names]

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
        # FIXME if the model has the same name as some other "real" module which we use, there will be trouble
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
