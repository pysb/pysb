from pysb.simulate import Simulator
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
    
def _translate(model, param_values=None, y0=None):
    gsp_model = gillespy.Model(model.name)
    gsp_model.add_parameter(_translate_parameters(model, param_values))
    gsp_model.add_species(_translate_species(model, y0))
    gsp_model.add_reaction(_translate_reactions(model))
    return gsp_model

class StochKitSimulator(Simulator):
        
    def __init__(self, model, tspan=None, cleanup=True, verbose=False):
        super(StochKitSimulator, self).__init__(model, tspan, verbose)
        generate_equations(self.model, cleanup, self.verbose)
    
    def run(self, tspan=None, param_values=None, y0=None, n_runs=1, seed=None, **additional_args):

        if tspan is not None:
            self.tspan = tspan
        elif self.tspan is None:
            raise Exception("'tspan' must be defined.")
        
        gsp_model = _translate(self.model, param_values, y0)
        trajectories = gillespy.StochKitSolver.run(gsp_model, t=(self.tspan[-1]-self.tspan[0]), number_of_trajectories=n_runs, \
                                                   increment=(self.tspan[1]-self.tspan[0]), seed=seed, **additional_args)
    
        # output time points (in case they aren't the same tspan, which is possible in BNG)
        self.tout = np.array(trajectories)[:,:,0] + self.tspan[0]
        # species
        self.y = np.array(trajectories)[:,:,1:]
        # observables and expressions
        self._calc_yobs_yexpr(param_values)
        
    def _calc_yobs_yexpr(self, param_values=None):
        
        self.yobs = []
        self.yobs_view = []
        self.yexpr = []
        self.yexpr_view = []
        
        # loop over simulations
        for n in range(len(self.y)):
            
            # observables
            if len(self.model.observables):
                self.yobs.append(np.ndarray(len(self.tout[n]), zip(self.model.observables.keys(), itertools.repeat(float))))
            else:
                self.yobs.append(np.ndarray((len(self.tout[n]), 0)))
            self.yobs_view.append(self.yobs[-1].view(float).reshape(len(self.yobs[-1]), -1))
            for i,obs in enumerate(self.model.observables):
                self.yobs_view[-1][:,i] = (self.y[n][:,obs.species] * obs.coefficients).sum(axis=1)
            
            # expressions
            exprs = self.model.expressions_dynamic()
            if len(exprs):
                self.yexpr.append(np.ndarray(len(self.tout[n]), zip(exprs.keys(), itertools.repeat(float))))
            else:
                self.yexpr.append(np.ndarray((len(self.tout[n]), 0)))
            self.yexpr_view.append(self.yexpr[-1].view(float).reshape(len(self.yexpr[-1]), -1))
            if not param_values:
                param_values = [p.value for p in self.model.parameters]
            p_subs = { p.name : param_values[i] for i,p in enumerate(self.model.parameters) }
            obs_names = self.model.observables.keys()
            obs_dict = { name : self.yobs_view[-1][:,i] for i,name in enumerate(obs_names) }
            for i,expr in enumerate(self.model.expressions_dynamic()):
                expr_subs = expr.expand_expr(self.model).subs(p_subs)
                func = sympy.lambdify(obs_names, expr_subs, "numpy")
                self.yexpr_view[-1][:,i] = func(**obs_dict)
            
        self.yobs = np.array(self.yobs)
        self.yobs_view = np.array(self.yobs_view)
        self.yexpr = np.array(self.yexpr)
        self.yexpr_view = np.array(self.yexpr_view)
    
    def get_yfull(self):
        
        sp_names = ['__s%d' % i for i in range(len(self.model.species))] 
        yfull_dtype = zip(sp_names, itertools.repeat(float))
        if len(self.model.observables):
            yfull_dtype += zip(self.model.observables.keys(), itertools.repeat(float))
        if len(self.model.expressions_dynamic()):
            yfull_dtype += zip(self.model.expressions_dynamic().keys(), itertools.repeat(float))
        yfull = []
        # loop over simulations
        for n in range(len(self.y)):
            yfull.append(np.ndarray(len(self.tout[n]), yfull_dtype))
            yfull_view = yfull[-1].view(float).reshape((len(yfull[-1]), -1))
            n_sp = self.y[n].shape[1]
            n_ob = self.yobs_view[n].shape[1]
            n_ex = self.yexpr_view[n].shape[1]
            yfull_view[:,:n_sp] = self.y[n]
            yfull_view[:,n_sp:n_sp+n_ob] = self.yobs_view[n]
            yfull_view[:,n_sp+n_ob:n_sp+n_ob+n_ex] = self.yexpr_view[n]
            
        yfull = np.array(yfull)
        return yfull

def run_stochkit(model, tspan, param_values=None, y0=None, n_runs=1, seed=None, verbose=False, **additional_args):

    sim = StochKitSimulator(model, verbose=verbose)
    sim.run(tspan, param_values, y0, n_runs, seed, **additional_args)
    yfull = sim.get_yfull()
    return sim.tout, yfull
    