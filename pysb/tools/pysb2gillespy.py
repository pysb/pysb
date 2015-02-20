import gillespy
import re
import sympy
import numpy as np
from pysb.bng import generate_equations
from numpy import rate

def _translate_parameters(model):
    param_list = []
    for p in model.parameters:
        found = False
        for u in model.parameters_unused():
            if p == u:
                found = True
        if not found:
            param_list.append(gillespy.Parameter(name=p.name, expression=p.value))
    return param_list

def _translate_species(model):
    species_list = []
    for i,sp in enumerate(model.species):
        found = False
        for ic in model.initial_conditions:
            if str(ic[0]) == str(sp):
                species_list.append(gillespy.Species(name="__s%d" % i,initial_value=np.round(ic[1].value)))
                found = True
        if not found:
            species_list.append(gillespy.Species(name="__s%d" % i,initial_value=0))
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
            rate = re.sub(r'\b%s\b' % e.name, '('+sympy.ccode(e.expand_expr())+')', rate)
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
        rxn_list.append(gillespy.Reaction(name = 'Rxn%d (rule:%s)' % (n, str(rxn["rule"])),\
                                          reactants = reactants,\
                                          products = products,\
                                          propensity_function = rate))        
    return rxn_list
    
def translate(model, verbose=False):
    generate_equations(model, verbose=verbose)
    gsp_model = gillespy.Model(model.name)
    gsp_model.add_parameter(_translate_parameters(model))
    gsp_model.add_species(_translate_species(model))
    gsp_model.add_reaction(_translate_reactions(model))
    return gsp_model
