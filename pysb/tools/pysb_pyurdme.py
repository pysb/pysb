try:
    import pyurdme
except ImportError:
    import warnings
    warnings.warn("Package 'pyurdme' cannot be found and is required for pyurdme simulations. See XXX for further details.")
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
            param_list[count] = pyurdme.Parameter(name=p.name, expression=val)
            count += 1
    return param_list

def _translate_species(model, initial_dist, dif0=None, y0=None):
    # Error check
    
   
    if y0 and len(y0) != len(model.species):
        raise Exception("len(dif0) and len(y0) must equal len(model.species)")            
    species_list = len(model.species) * [None]
    for i,sp in enumerate(model.species):
        val = 0.
        if dif0:
            val=dif0[i]
        else:
            for id in model.diffusivities:
                if str(id[0]) == str(sp):
                    val=id[1]
        species_list[i] = pyurdme.Species(name="__s%d" % i, diffusion_constant=val)

    distribution_list = len(initial_dist.keys()) * [None]
    for i,sp in enumerate(model.species):
        if str(sp) in initial_dist.keys():
            val = 0.
            if y0:
                val=y0[i]
            else:
                for ic in model.initial_conditions:
                    if str(ic[0]) == str(sp):
                        val=np.round(ic[1].value)
            distribution_list[i] = [initial_dist[str(sp)],{'__s%d'%i:val}]
        else: pass

    return species_list, distribution_list

    
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
        rxn_list[n] = pyurdme.Reaction(name = 'rule%s' %  str(rxn["rule"]),\
                                    reactants = reactants,\
                                    products = products,\
                                    propensity_function = rate)
    return rxn_list
    
def _translate(model, mesh, initial_dist, param_values=None, dif0=None, y0=None):
    
    
    urdme_model = pyurdme.URDMEModel(model.name)
    urdme_model.add_species(_translate_species(model, initial_dist, dif0, y0)[0])
    urdme_model.mesh = mesh
    urdme_model.add_parameter(_translate_parameters(model, param_values))
    urdme_model.add_reaction(_translate_reactions(model))
    initial_d_info = _translate_species(model, initial_dist, dif0, y0)[1]
    for id in initial_d_info:
        getattr(urdme_model, id[0][0])({urdme_model.get_species(id[1].keys()[0]):id[1].values()[0]}, id[0][1])
       
    return urdme_model

class PyurdmeSimulator(Simulator):
        
    def __init__(self, model, tspan=None, mesh=None, initial_dist=None, cleanup=True, verbose=False):
        super(PyurdmeSimulator, self).__init__(model, tspan, verbose)
        generate_equations(self.model, cleanup, self.verbose)
    
    def run(self, tspan=None, mesh = None, initial_dist=None, param_values=None, dif0=None, y0=None, solver = 'nsm'):


        if tspan is not None:
            self.tspan = tspan
        elif self.tspan is None:
            raise Exception("'tspan' must be defined.")
        
        if mesh is not None:
            self.mesh = mesh
        elif self.mesh is None:
            raise Exception("Mesh must be defined.")
        
        if initial_dist is not None:
            self.initial_dist = initial_dist
        elif self.initial_dist is None:
            raise Exception("Initial distribution of proteins must be defiened")        
        
        
        urdme_model = _translate(self.model, self.mesh, self.initial_dist, param_values, dif0, y0)
        urdme_model.timespan(self.tspan)
        
        result = urdme_model.run(report_level=1)
    
        # species
        
        trajectories = np.zeros((len(result['sol'].keys()),len(result.get_timespan())))
        for i, sp in enumerate(result['sol'].keys()):
            trajectories[i] = np.sum(result.get_species(sp),axis=1)
        trajectories = trajectories.T
        
        self.y = trajectories
        
        # output time points (in case they aren't the same tspan, which is possible in BNG)
        time = result.get_timespan()
        self.tout = np.tile(time,(len(self.y),1))

        # observables and expressions
        self._calc_yobs_yexpr(param_values)
        
        self.simulation = result
        
    def _calc_yobs_yexpr(self, param_values=None):
        super(PyurdmeSimulator, self)._calc_yobs_yexpr()
        
    def get_yfull(self):
        return super(PyurdmeSimulator, self).get_yfull()        


def run_pyurdme(model, tspan, mesh, initial_dist, param_values=None, dif0=None, y0=None, verbose=True):
    """Runs pyurdme using PySB
       the initial distribution should be a dict:
       initial_dist={DISC(bf=None):['set_initial_condition_scatter', *arguments, i.e point=[0.5,0.5]]}"""
    sim = PyurdmeSimulator(model, verbose=verbose)
    sim.run(tspan, mesh, initial_dist, param_values , dif0, y0)
    return sim.simulation
