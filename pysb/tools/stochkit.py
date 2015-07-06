try:
    import gillespy
except:
    import warnings
    warnings.warn("Package 'gillespy' cannot be found and is required for StochKit simulations. See XXX for further details.")
from pysb.simulate import Simulator
from pysb.bng import generate_equations
import re
import sympy
import numpy as np
import itertools
import pysb

def _translate_parameters(model, param_values=None):
    # Error check
    if param_values is not None and len(param_values) != len(model.parameters):
        raise Exception("len(param_values) must equal len(model.parameters)")
    unused = model.parameters_unused()
    param_list = (len(model.parameters)-len(unused)) * [None]
    count = 0
    for i,p in enumerate(model.parameters):
        if p not in unused:
            if param_values is not None:
                val=param_values[i]
            else:
                val=p.value
            param_list[count] = gillespy.Parameter(name=p.name, expression=val)
            count += 1
    return param_list

def _translate_species(model, y0=None):
    # Error check
    if y0 and len(y0) != len(model.species):
        raise Exception("len(y0) must equal len(model.species)")            
    species_list = len(model.species) * [None]
    for i,sp in enumerate(model.species):
        val = 0.
        if y0:
            val=y0[i]
        else:
            for ic in model.initial_conditions:
                if str(ic[0]) == str(sp):
                    val=np.round(ic[1].value)
        species_list[i] = gillespy.Species(name="__s%d" % i,initial_value=val)
    return species_list
    
def _translate_reactions(model):
    rxn_list = len(model.reactions) * [None]
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
        # replace terms like __s**2 with __s*(__s-1)
        rate = str(rxn["rate"])
        pattern = "(__s\d+)\*\*(\d+)"
        matches = re.findall(pattern, rate)
        for m in matches:
            repl = m[0]
            for i in range(1,int(m[1])):
                repl += "*(%s-%d)" % (m[0],i)
            rate = re.sub(pattern, repl, rate)
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
        rxn_list[n] = gillespy.Reaction(name = 'Rxn%d (rule:%s)' % (n, str(rxn["rule"])),\
                                    reactants = reactants,\
                                    products = products,\
                                    propensity_function = rate)
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
        super(StochKitSimulator, self)._calc_yobs_yexpr()
        
    def get_yfull(self):
        return super(StochKitSimulator, self).get_yfull()

def run_stochkit(model, tspan, param_values=None, y0=None, n_runs=1, seed=None, verbose=False, **additional_args):

    sim = StochKitSimulator(model, verbose=verbose)
    sim.run(tspan, param_values, y0, n_runs, seed, **additional_args)
    yfull = sim.get_yfull()
    return sim.tout, yfull
    
