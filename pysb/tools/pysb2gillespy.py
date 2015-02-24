import gillespy
from pysb.bng import generate_equations
import re
import sympy
import numpy as np
import itertools

def _translate_parameters(model, param_values=None):
    # Error check
    if param_values and len(param_values) != len(model.parameters):
        raise Exception("len(param_values) must equal len(model.parameters)")
    param_list = []
    unused = model.parameters_unused()
    for i,p in enumerate(model.parameters):
        if p not in unused:
            if param_values:
                val=param_values[i]
            else:
                val=p.value
            param_list.append(gillespy.Parameter(name=p.name, expression=val))
    return param_list

def _translate_species(model, y0=None):
    # Error check
    if y0 and len(y0) != len(model.species):
        raise Exception("len(y0) must equal len(model.species)")            
    species_list = []
    for i,sp in enumerate(model.species):
        val = 0.
        if y0:
            val=y0[i]
        else:
            for ic in model.initial_conditions:
                if str(ic[0]) == str(sp):
                    val=np.round(ic[1].value)
        species_list.append(gillespy.Species(name="__s%d" % i,initial_value=val))
    return species_list
    
def _translate_reactions(model):
    rxn_list = []
    for n,rxn in enumerate(model.reactions):
        reactants = {}
        products = {}
        # reactants        
        for r in rxn["reactants"]:
            r = "__s%d" % r
            if r in reactants:
                reactants[r] += 1
            else:
                reactants[r] = 1
        # products
        for p in rxn["products"]:
            p = "__s%d" % p
            if p in products:
                products[p] += 1
            else:
                products[p] = 1
        rate = sympy.ccode(rxn["rate"])
        # expand expressions
        for e in model.expressions:
            rate = re.sub(r'\b%s\b' % e.name, '('+sympy.ccode(e.expand_expr(model))+')', rate)
        # replace observables w/ sums of species
        for obs in model.observables:
            obs_string = ''
            for i in range(len(obs.coefficients)):
                if i > 0: obs_string += "+"
                if obs.coefficients[i] > 1: 
                    obs_string += str(obs.coefficients[i])+"*"
                obs_string += "__s"+str(obs.species[i])
            if len(obs.coefficients) > 1: 
                obs_string = '(' + obs_string + ')'
            rate = re.sub(r'%s' % obs.name, obs_string, rate)
        # create reaction
        rxn_list.append(gillespy.Reaction(name = 'Rxn%d (rule:%s)' % (n, str(rxn["rule"])),\
                                          reactants = reactants,\
                                          products = products,\
                                          propensity_function = rate))        
    return rxn_list
    
def _translate(model, param_values=None, y0=None, verbose=False):
    gsp_model = gillespy.Model(model.name)
    gsp_model.add_parameter(_translate_parameters(model, param_values))
    gsp_model.add_species(_translate_species(model, y0))
    gsp_model.add_reaction(_translate_reactions(model))
    return gsp_model

def run_stochkit(model, tspan, param_values=None, y0=None, n_runs=1, seed=None, verbose=False, **additional_args):
    
    generate_equations(model, verbose=verbose)
    gsp_model = _translate(model, param_values=None, y0=None, verbose=verbose)
    trajectories = gillespy.StochKitSolver.run(gsp_model, t=(tspan[-1]-tspan[0]), number_of_trajectories=n_runs, increment=(tspan[1]-tspan[0]), seed=seed, **additional_args)
    
    # create output array dtype
    names = ['time'] + ['__s%d' % i for i in range(len(model.species))] 
    yfull_dtype = zip(names, itertools.repeat(float))
    if len(model.observables):
        yfull_dtype += zip(model.observables.keys(), itertools.repeat(float))
    if len(model.expressions_dynamic()):
        yfull_dtype += zip(model.expressions_dynamic().keys(), itertools.repeat(float))

    # loop over simulations
    yfull = []
    for n in range(n_runs):
        
        yfull.append(np.ndarray(len(tspan), yfull_dtype))
        yfull_view = yfull[-1].view(float).reshape((len(tspan), -1))
        
        # time
        yfull_view[:,0] = trajectories[n][:,0]

        # species
        yfull_view[:,1:1+len(model.species)] = trajectories[n][:,1:]
        
        # calculate observables
        for i,obs in enumerate(model.observables):
            index = 1+len(model.species)+i
            obs_species = np.array(obs.species)+1 # element 0 is time
            yfull_view[:,index] = (trajectories[n][:,obs_species] * obs.coefficients).sum(axis=1)
            
        # calculate expressions
        if not param_values:
            param_values = [p.value for p in model.parameters]
        p_subs = { p.name : param_values[i] for i,p in enumerate(model.parameters) }
        obs_names = model.observables.keys()
        obs_dict = { name : yfull_view[:,1+len(model.species)+i] for i,name in enumerate(obs_names) }
        for i,expr in enumerate(model.expressions_dynamic()):
            index = 1+len(model.species)+len(model.observables)+i
            expr_subs = expr.expand_expr(model).subs(p_subs)
            func = sympy.lambdify(obs_names, expr_subs, "numpy")
            yfull_view[:,index] = func(**obs_dict)
        
    yfull = np.array(yfull)
    return yfull
        