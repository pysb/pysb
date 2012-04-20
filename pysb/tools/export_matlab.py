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

  output.write("% MATLAB model definition file\n")
  output.write('%% save as %s_odes.m\n' % model_name)
  output.write('function out = %s_odes(t, input, param)' % model_name)
  output.write("\n\n")

  c_code_consts = '\n'.join(['param(%d) = %s %% %s;' % (i+1, p.value, p.name) for i, p in
      enumerate(model.parameters)])
  c_code_eqs = '\n'.join(['out(%d,1) = %s;' % (i+1, sympy.ccode(model.odes[i])) for i in
      range(len(model.odes))])
  c_code_eqs = re.sub(r's(\d+)', lambda m: 'input(%s)' % (int(m.group(1))+1), c_code_eqs)
  c_code_eqs = re.sub(r'pow\(', 'power(', c_code_eqs)
  c_code = c_code_eqs

  c_code_species = '\n'.join(['%% input(%d) = %s;' % (i+1, s) for i, s in
      enumerate(model.species)])

  for i, p in enumerate(model.parameters):
    c_code = re.sub(r'\b(%s)\b' % p.name, 'param(%d)' % (i+1), c_code)

  output.write(c_code_consts + "\n\n")
  output.write(c_code_species + "\n\n")
  output.write(c_code + "\n\n")
  output.write("end\n")
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



