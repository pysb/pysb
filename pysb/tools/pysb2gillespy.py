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

    # time (just in case they aren't the same for all sims -- possible in BNG)
    t = np.array(trajectories)[:,:,0]
    
    # species
    y = np.array(trajectories)[:,:,1:]

    # loop over simulations
    yobs = []
    yobs_view = []
    yexpr = []
    yexpr_view = []
    for n in range(n_runs):
        
        # observables
        if len(model.observables):
            yobs.append(np.ndarray(len(t[n]), zip(model.observables.keys(), itertools.repeat(float))))
        else:
            yobs.append(numpy.ndarray((len(t[n]), 0)))
        yobs_view.append(yobs[-1].view(float).reshape(len(yobs[-1]), -1))
        for i,obs in enumerate(model.observables):
            yobs_view[-1][:,i] = (y[n,:,obs.species] * obs.coefficients).sum(axis=1)
        
        # expressions
        exprs = model.expressions_dynamic()
        if len(exprs):
            yexpr.append(numpy.ndarray(len(t[n]), zip(exprs.keys(), itertools.repeat(float))))
        else:
            yexpr.append(np.ndarray((len(t[n]), 0)))
        yexpr_view.append(yexpr[-1].view(float).reshape(len(yexpr[-1]), -1))
        if not param_values:
            param_values = [p.value for p in model.parameters]
        p_subs = { p.name : param_values[i] for i,p in enumerate(model.parameters) }
        obs_names = model.observables.keys()
        obs_dict = { name : yobs_view[-1][:,i] for i,name in enumerate(obs_names) }
        for i,expr in enumerate(model.expressions_dynamic()):
            expr_subs = expr.expand_expr(model).subs(p_subs)
            func = sympy.lambdify(obs_names, expr_subs, "numpy")
            yexpr_view[-1][:,i] = func(**obs_dict)
        
    yobs = np.array(yobs)
    yobs_view = np.array(yobs_view)
    yexpr = np.array(yexpr)
    yexpr_view = np.array(yexpr_view)
    
    # full output
    sp_names = ['__s%d' % i for i in range(len(model.species))] 
    yfull_dtype = zip(sp_names, itertools.repeat(float))
    if len(model.observables):
        yfull_dtype += zip(model.observables.keys(), itertools.repeat(float))
    if len(model.expressions_dynamic()):
        yfull_dtype += zip(model.expressions_dynamic().keys(), itertools.repeat(float))
    yfull = []
    for n in range(n_runs):
        yfull.append(np.ndarray(len(t[n]), yfull_dtype))
        yfull_view = yfull[-1].view(float).reshape((len(yfull[-1]), -1))
        n_sp = y[n].shape[1]
        n_ob = yobs_view[n].shape[1]
        n_ex = yexpr_view[n].shape[1]
        yfull_view[:,:n_sp] = y[n]
        yfull_view[:,n_sp:n_sp+n_ob] = yobs_view[n]
        yfull_view[:,n_sp+n_ob:n_sp+n_ob+n_ex] = yexpr_view[n]
        
    yfull = np.array(yfull)

    return t, yfull
    