#!/usr/bin/env python

import pysb
import pysb.bng
import sympy
import re
import sys
import os
from StringIO import StringIO
from math import floor, log

def run(model):
  output = StringIO()
  pysb.bng.generate_equations(model)

  output.write("(* Mathematica model definition file *)\n")
  output.write("(* Model Name: " + model.name + " *)\n\n")
  output.write("(*\n")
  output.write("Run with, for example:\n")
  output.write("tmax = 10\n")
  output.write("soln = NDSolve[Join[eqns, initconds], slist, {t, 0, tmax}]\n")
  output.write("Plot[s0[t] /. soln, {t, 0, tmax}, PlotRange -> All]\n")
  output.write("*)\n\n")

  #output.write('(* save as %s_odes.txt *)\n' % model_name)
  #output.write("\n\n")

  # PARAMETERS
  # Note that in Mathematica, underscores are not allowed in variable names, so
  # we simply strip them out here
  c_code_consts = '' 
  for i, p in enumerate(model.parameters):
    pname = p.name.replace('_', '')
    exp = floor(log(p.value, 10))

    if (not exp == 0):
      c_code_consts += '%s = %g * 10^%g;\n' %  (pname, p.value / 10**exp, exp)
    else:
      c_code_consts += '%s = %g;\n' %  (pname, p.value)

  ## ODEs ###
  c_code_eqs = 'eqns = {\n'
  # Concatenate the equations
  c_code_eqs += ',\n'.join(['s%d == %s' % (i, sympy.ccode(model.odes[i]))
                           for i in range(len(model.odes))])
  # Replace, e.g., s0 with s[0]
  c_code_eqs = re.sub(r's(\d+)', lambda m: 's%s[t]' % (int(m.group(1))),
                      c_code_eqs)
  # Add the derivative symbol ' to the left hand sides
  c_code_eqs = re.sub(r's(\d+)\[t\] ==', r"s\1'[t] ==", c_code_eqs)
  # Correct the exponentiation syntax
  c_code_eqs = re.sub(r'pow\((.*), (.*)\)', r'\1^\2', c_code_eqs)
  c_code_eqs += '\n}'
  #c_code = c_code_eqs
  # Eliminate underscores from parameter names in equations
  for i, p in enumerate(model.parameters):
    c_code_eqs = re.sub(r'\b(%s)\b' % p.name, p.name.replace('_', ''), c_code_eqs)

  ## INITIAL CONDITIONS
  ic_values = ['0'] * len(model.odes)
  for i, ic in enumerate(model.initial_conditions):
    ic_values[model.get_species_index(ic[0])] = ic[1].name.replace('_', '')

  c_code_ics = 'initconds = {\n'
  c_code_ics += ',\n'.join(['s%s[0] == %s' % (i, val) for i, val in enumerate(ic_values)])
  c_code_ics += '\n}'
  
  ## SOLVE LIST
  c_code_slist = 'slist = {\n'
  c_code_slist += ',\n'.join(['s%s[t]' % (i) for i in range(0, len(model.odes))])
  c_code_slist += '\n}'

  ## OBSERVABLES
  c_code_obs = ''
  for obs_name in model.observable_groups:
    c_code_obs += obs_name + ' = '
    groups = model.observable_groups[obs_name]
    c_code_obs += '+'.join(['s%s[t]' % (g[1]) for g in groups]) 
    c_code_obs += ' /. soln\n' 
  
  # Add comments identifying the species
  c_code_species = '\n'.join(['(* s%d[t] = %s *)' % (i, s) for i, s in
                        enumerate(model.species)])

  output.write('(* Parameters *)\n')
  output.write(c_code_consts + "\n\n")
  output.write('(* List of Species *)\n')
  output.write(c_code_species + "\n\n")
  output.write('(* ODEs *)\n')
  output.write(c_code_eqs + "\n\n")
  output.write('(* Initial Conditions *)\n')
  output.write(c_code_ics + "\n\n")
  output.write('(* List of Variables (e.g., as an argument to NDSolve) *)\n')
  output.write(c_code_slist + '\n\n')
  output.write('(* Run the simulation *)\n')
  output.write('tmax = 100\n')
  output.write('soln = NDSolve[Join[eqns, initconds], slist, {t, 0, tmax}]\n\n')
  output.write('(* Observables *)\n')
  output.write(c_code_obs + '\n\n')

  #output.write("end\n")
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



