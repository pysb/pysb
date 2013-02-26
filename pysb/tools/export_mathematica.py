#!/usr/bin/env python
"""
A module for converting a PySB model to a set of ordinary differential
equations for integration in Mathematica. Can be used as a command-line script
or from within the Python shell.
"""

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

    # Header comment
    output.write("(* Mathematica model definition file *)\n")
    output.write("(* Model Name: " + model.name + " *)\n\n")
    output.write("(*\n")
    output.write("Run with, for example:\n")
    output.write("tmax = 10\n")
    output.write("soln = NDSolve[Join[eqns, initconds], slist, {t, 0, tmax}]\n")
    output.write("Plot[s0[t] /. soln, {t, 0, tmax}, PlotRange -> All]\n")
    output.write("*)\n\n")

    # PARAMETERS
    # Note that in Mathematica, underscores are not allowed in variable names,
    # so we simply strip them out here
    params_str = ''
    for i, p in enumerate(model.parameters):
        pname = p.name.replace('_', '')

        # Convert parameter values to scientific notation
        # If the parameter is 0, don't take the log!
        if p.value == 0:
            params_str += '%s = %g;\n' %  (pname, p.value)
        # Otherwise, take the log (base 10) and format accordingly
        else:
            exp = floor(log(p.value, 10))
            if (not exp == 0):
                params_str += '%s = %g * 10^%g;\n' % \
                        (pname, p.value / 10**exp, exp)
            else:
                params_str += '%s = %g;\n' %  (pname, p.value)

    ## ODEs ###
    odes_str = 'eqns = {\n'
    # Concatenate the equations
    odes_str += ',\n'.join(['s%d == %s' % (i, sympy.ccode(model.odes[i]))
                           for i in range(len(model.odes))])
    # Replace, e.g., s0 with s[0]
    odes_str = re.sub(r's(\d+)', lambda m: 's%s[t]' % (int(m.group(1))),
                      odes_str)
    # Add the derivative symbol ' to the left hand sides
    odes_str = re.sub(r's(\d+)\[t\] ==', r"s\1'[t] ==", odes_str)
    # Correct the exponentiation syntax
    odes_str = re.sub(r'pow\((.*), (.*)\)', r'\1^\2', odes_str)
    odes_str += '\n}'
    #c_code = odes_str
    # Eliminate underscores from parameter names in equations
    for i, p in enumerate(model.parameters):
        odes_str = re.sub(r'\b(%s)\b' % p.name, p.name.replace('_', ''),
                          odes_str)

    ## INITIAL CONDITIONS
    ic_values = ['0'] * len(model.odes)
    for i, ic in enumerate(model.initial_conditions):
        ic_values[model.get_species_index(ic[0])] = ic[1].name.replace('_', '')

    init_conds_str = 'initconds = {\n'
    init_conds_str += ',\n'.join(['s%s[0] == %s' % (i, val)
                                 for i, val in enumerate(ic_values)])
    init_conds_str += '\n}'

    ## SOLVE LIST
    solvelist_str = 'solvelist = {\n'
    solvelist_str += ',\n'.join(['s%s[t]' % (i)
                                for i in range(0, len(model.odes))])
    solvelist_str += '\n}'

    ## OBSERVABLES
    observables_str = ''
    for obs in model.observables:
        observables_str += obs.name + ' = '
        #groups = model.observable_groups[obs_name]
        observables_str += ' + '.join(['(s%s[t] * %d)' % (s, c)
                              for s, c in zip(obs.species, obs.coefficients)])
        observables_str += ' /. soln\n' 

    # Add comments identifying the species
    species_str = '\n'.join(['(* s%d[t] = %s *)' % (i, s) for i, s in
                        enumerate(model.species)])

    output.write('(* Parameters *)\n')
    output.write(params_str + "\n")
    output.write('(* List of Species *)\n')
    output.write(species_str + "\n\n")
    output.write('(* ODEs *)\n')
    output.write(odes_str + "\n\n")
    output.write('(* Initial Conditions *)\n')
    output.write(init_conds_str + "\n\n")
    output.write('(* List of Variables (e.g., as an argument to NDSolve) *)\n')
    output.write(solvelist_str + '\n\n')
    output.write('(* Run the simulation -- example *)\n')
    output.write('tmax = 100\n')
    output.write('soln = NDSolve[Join[eqns, initconds], ' \
                 'solvelist, {t, 0, tmax}]\n\n')
    output.write('(* Observables *)\n')
    output.write(observables_str + '\n\n')

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



