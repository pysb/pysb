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

    ## PARAMETERS
    param_list = '\n'.join(['param(%d) = %s; %% %s' % (i+1, p.value, p.name) for i, p in
       enumerate(model.parameters)])

    param_list_file = open('%s_nominal_values.m' % model.name, 'w')
    param_list_file.write('function param = %s_nominal_values()\n\n' % model.name)
    param_list_file.write(param_list)
    param_list_file.write('\n\nend\n')
    param_list_file.close()

    ## ODES
    odes = 'function out = %s_odes(t, input, param)\n\n' % model.name

    species_list = '\n'.join(['%% input(%d) = %s;' % (i+1, s) for i, s in
        enumerate(model.species)])
    odes += species_list + "\n\n"

    odes += '\n'.join(['%% %s\nout(%d,1) = %s;' % (model.species[i], i+1, sympy.ccode(model.odes[i]))
                       for i in range(len(model.odes))])
    odes = re.sub(r's(\d+)', lambda m: 'input(%s)' % (int(m.group(1))+1), odes)
    odes = re.sub(r'pow\(', 'power(', odes)
    odes += '\n\nend\n'

    # Substitute the parameter references into the ODE expressions
    for i, p in enumerate(model.parameters):
        odes = re.sub(r'\b(%s)\b' % p.name, 'param(%d)' % (i+1), odes)

    ode_file = open('%s_odes.m' % model.name, 'w')
    ode_file.write(odes)
    ode_file.close()

    ## OBSERVABLES
    ## -- hoda
    # function [ode_observables, kd_values, kd_index, ic_index, dividing_factor] = %s_observables()
    # ode_observables is a cell array, with three entries for each index: first is a list of species
    # second is a matched list of coefficients, and the third is a normalization factor (usually an
    # condition) or 1 if not needed
    #
    # kd_values is a cell array where each entry is a tuple of the relevant initial conditions
    # kd_index is a matched cell array with each entry a tuple for the index of the relevant parameters to
    # be perturbed in their initial conditions
    # ic_index=cell array with a list that is mapped to the ODEs (i.e., the init_conds) file
    observables = "function [ode_observables, kd_values, kd_index, "
    observables += "ic_index, dividing_factor] = %s_observables()\n\n" % model.name

    for i, obs in enumerate(model.observables):
        obs_spec_list = "[" + ' '.join(str(s+1) for s in obs.species) + "]"
        obs_coeff_list = "[" + ' '.join(str(c) for c in obs.coefficients) + "]"

        observables += "ode_observables{%d, 1} = %s; %% %s\n" % (i+1, obs_spec_list, str(obs))
        observables += "ode_observables{%d, 2} = %s;\n" % (i+1, obs_coeff_list)
        observables += "ode_observables{%d, 3} = 1;\n" % (i+1)
    observables += "end\n"

    obs_file = open('%s_observables.m' % model.name, 'w')
    obs_file.write(observables)
    obs_file.close()

    ## PRIOR FLAGS
    # function prior_flag=%s_prior_flags()
    # returns array: if the param is an initial condition, set it to 0 in the array
    # otherwise: returns 1 (future implementation to return a value for
    # 1st forsrd, 1st, reverse, 2nd order forward, (2nd reverse), catalytic (1-5)
    # for now default to 1
    ic_names = [ic[1].name for ic in model.initial_conditions]

    prior_flags = "function prior_flag=%s_prior_flags()\n\n" % model.name
    for i, p in enumerate(model.parameters):    
        if (p.name in ic_names):
            prior_flags += "prior_flag(%d) = 0; %% %s\n" % (i+1, p.name)
        else:
            prior_flags += "prior_flag(%d) = 1; %% %s\n" % (i+1, p.name)
    prior_flags += "\nend\n" 

    prior_flags_file = open('%s_prior_flags.m' % model.name, 'w')
    prior_flags_file.write(prior_flags)
    prior_flags_file.close()
 
    ## INITIAL CONDITIONS
    # function conc = %s_init_conds()
    # returns an array that matches the num of ODEs (mostly zeros)
    init_conds = "function conc = %s_init_conds()\n\n" % model.name
    init_conds += "conc = zeros(%d, 1);\n" % len(model.species)
    for ic in model.initial_conditions:
        init_conds += 'conc(%d) = %g; %% %s; %s\n' % (model.get_species_index(ic[0])+1, ic[1].value,
                                                  ic[1].name, ic[0])
    init_conds += "\nend\n\n"

    init_conds_file = open('%s_init_conds.m' % model.name, 'w')
    init_conds_file.write(init_conds)
    init_conds_file.close()

    #output.write(param_list + "\n\n")
    #output.write(c_code_species + "\n\n")
    #output.write(odes + "\n\n")
    #output.write("end\n")
    #return output.getvalue()

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
    # print run(model)
    run(model)



