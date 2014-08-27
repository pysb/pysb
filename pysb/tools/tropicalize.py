import networkx
import sympy
import re
import copy
import numpy
import sympy.parsing.sympy_parser
from pysb.integrate import odesolve
from collections    import Mapping

# Helper class to use evalf with a ndarray
# class FilterNdarray(Mapping):
#     def __init__(self, source):
#         self.source = source
#         self.t = 0
# 
#     def __getitem__(self, key):
#         # WARNING: This is a monkey patch when evalf sends a number to getitem instead of a symbol
#         try:
#             return self.source[key.name][self.t]
#         except:
#             return key.name
# 
#     def __len__(self): return len(self.source)
# 
#     def __iter__(self):
#         for key in self.source:
#             yield key
# 
#     def set_time(self, t):
#         self.t = t
#         return self

class Tropical:
    def __init__(self, model):
        self.model            = model
        self.t                = numpy.linspace(0, 100, 10001) # timerange used
        self.y                = None # odes solution, numpy array
        self.slaves           = None
        self.graph            = None
        self.cycles           = []
        self.conservation     = None
        self.conserve_var     = None
        self.full_names       = {}

    def __repr__(self):
        return "<%s '%s' (slaves: %s, cycles: %d) at 0x%x>" % \
            (self.__class__.__name__, self.model.name,
             self.slaves.__repr__(),
             len(self.cycles),
             id(self))

    def tropicalize(self, ignore=10, epsilon=0.0001, rho=1, verbose=True):
        if verbose: print "Solving Simulation"
        self.y = odesolve(self.model, self.t)
    
        # Only concrete species are considered, and the names must be made to match
        names           = [n for n in filter(lambda n: n.startswith('__'), self.y.dtype.names)]
        self.y          = self.y[names]
        names           = [n.replace('__','') for n in names]
        self.y.dtype    = [(n,'<f8') for n in names]
    
        if verbose: print "Getting slaved species"
        self.find_slaves(self.y[ignore:], verbose, epsilon)
        if verbose: print "Constructing Graph"
        self.construct_graph()
        if verbose: print "Computing Cycles"
        self.cycles = list(networkx.simple_cycles(self.graph))
        if verbose: print "Computing Conservation laws"
        (self.conservation, self.conserve_var) = self.mass_conserved()
        if verbose: print "Pruning Equations"
        self.pruned = self.pruned_equations(self.y[ignore:], rho)

# FIXME: THIS STEP IS BROKEN DUE TO THE ADDITION OF CYCLES
        #if verbose: print "Solving pruned equations"
        #self.sol_pruned = self.solve_pruned()
        
        return self

    # Compute imposed distances of a model
    def find_slaves(self, y, verbose=False, epsilon=None):
        distances = []
        self.slaves = []
        a = [] #list of solved polynomial equations
        b = []

        # Loop through all equations (i is equation number)
        for i, eq in enumerate(self.model.odes):
            eq        = eq.subs('s%d' % i, 's%dstar' % i)
            sol       = sympy.solve(eq, sympy.Symbol('s%dstar' % i)) # Find equation of imposed trace
            for j in range(len(sol)):  # j is solution j for equation i (mostly likely never greater than 2)
                for p in self.model.parameters: sol[j] = sol[j].subs(p.name, p.value) # Substitute parameters
                a.append(sol[j])
                b.append(i)
        for k,e in enumerate(a):
            args = [] #arguments to put in lambdify function
            variables = [atom for atom in a[k].atoms(sympy.Symbol) if not re.match(r'\d',str(atom))]
            f = sympy.lambdify(variables, a[k], modules = dict(sqrt=numpy.lib.scimath.sqrt) )
            variables_l = list(variables)
       # print variables
            for u,l in enumerate(variables_l):
                args.append(y[:][str(l)])
            hey = abs(f(*args) - y[:]['s%d'%b[k]])
            if hey.max() <= epsilon : self.slaves.append(b[k])
        #print hey.max()                
                
#                 

        return self.slaves


    # This is a function which builds the edges according to the nodes
    def r_link(self, graph, s, r, **attrs):
        nodes = (s, r)
        if attrs.get('_flip'):
            del attrs['_flip']
            nodes = reversed(nodes)
        attrs.setdefault('arrowhead', 'normal')
        graph.add_edge(*nodes, **attrs)

    def construct_graph(self):
        if(self.model.odes == None or self.model.odes == []):
            pysb.bng.generate_equations(model)

        self.graph = networkx.DiGraph(rankdir="LR")
        ic_species = [cp for cp, parameter in self.model.initial_conditions]
        for i, cp in enumerate(self.model.species):
            species_node = i
            self.graph.add_node(species_node, label=species_node)
        for i, reaction in enumerate(self.model.reactions):       
            reactants = set(reaction['reactants'])
            products = set(reaction['products']) 
            attr_reversible = {}
            for s in reactants:
                for p in products:
                    self.r_link(self.graph, s, p, **attr_reversible)
        return self.graph

    #This function finds conservation laws from the conserved cycles
    def mass_conserved(self, verbose=False):
        if(self.model.odes == None or self.model.odes == []):
            pysb.bng.generate_equations(self.model)
        h = [] # Array to hold conservation equation
        g = [] # Array to hold corresponding lists of free variables in conservation equations
        for i, item in enumerate(self.cycles):
            b = 0
            u = 0
            for j, specie in enumerate(item):
                b += self.model.odes[self.cycles[i][j]]
            if b == 0:
                g.append(item)
                for l,k in enumerate(item):
                    u += sympy.Symbol('s%d' % self.cycles[i][l])    
                h.append(u-sympy.Symbol('C%d'%i))
                if verbose: print '  cycle%d'%i, 'is conserved'
        (self.conservation, self.conserve_var) = h,g
        return h, g

    def slave_equations(self):
        if(self.model.odes == None or self.model.odes == []):
            eq = self.model.odes
        slave_conserved_eqs = {}
        for i, j in enumerate(self.slaves):
            slave_conserved_eqs[j] = self.model.odes[self.slaves[i]]
        return slave_conserved_eqs

    def find_nearest_zero(self, array):
        idx = (numpy.abs(array)).argmin()
        return array[idx]

    # Make sure this is the "ignore:" y
    def pruned_equations(self, y, rho=1):
        pruned_eqs = self.slave_equations()
        eqs        = copy.deepcopy(pruned_eqs)

        for i, j in enumerate(eqs):
            ble = eqs[j].as_coefficients_dict().keys() # Get monomials
            for l, m in enumerate(ble): #Compares the monomials to find the pruned system
                m_ready = m # Monomial to compute with
                m_elim  = m # Monomial to save
                for p in self.model.parameters: m_ready = m_ready.subs(p.name, p.value) # Substitute parameters
                for k in range(len(ble)):
                    if (k+l+1) <= (len(ble)-1):
                        ble_ready = ble[k+l+1] # Monomial to compute with
                        ble_elim  = ble[k+l+1] # Monomial to save
                        for p in self.model.parameters: ble_ready = ble_ready.subs(p.name, p.value) # Substitute parameters
                        args2 = []
                        args1 = []
                        variables_ble_ready = [atom for atom in ble_ready.atoms(sympy.Symbol) if not re.match(r'\d',str(atom))]
                        variables_m_ready = [atom for atom in m_ready.atoms(sympy.Symbol) if not re.match(r'\d',str(atom))]
                        f_ble = sympy.lambdify(variables_ble_ready, ble_ready, 'numpy' )
                        f_m = sympy.lambdify(variables_m_ready, m_ready, 'numpy' )
                        for uu,ll in enumerate(variables_ble_ready):
                            args2.append(y[:][str(ll)])
                        for w,s in enumerate(variables_m_ready):
                            args1.append(y[:][str(s)])
                        hey_pruned = f_m(*args1) - f_ble(*args2)
                        diff = self.find_nearest_zero(hey_pruned)
                        diff_pru = numpy.abs(diff)
                        if diff > 0 and diff_pru > rho:
                            pruned_eqs[j] = pruned_eqs[j].subs(ble_elim, 0)
                        if diff < 0 and diff_pru > rho:\
                            pruned_eqs[j] = pruned_eqs[j].subs(m_elim, 0)   
                            
        for i, l in enumerate(self.conservation): #Add the conservation laws to the pruned system
            pruned_eqs['cons%d'%i]=l
        self.pruned = pruned_eqs
        return pruned_eqs

    def solve_pruned(self):
        solve_for = copy.deepcopy(self.slaves)
        eqs       = copy.deepcopy(self.pruned)

        # Locate single protein conserved (s2 in tyson, the solver now knows what is constant)
        for i in self.conserve_var:
            if len(i) == 1:
                solve_for.append(i[0])
        variables =  [sympy.Symbol('s%d' %var) for var in solve_for ]
        sol = sympy.solve_poly_system(eqs, variables)
        
        # This if 'effed right here! @$%#@$%@#$%@#$%!!!!
        self.sol_pruned = { j:sol[0][i] for i, j in enumerate(solve_for) }
        return self.sol_pruned

    def equations_to_tropicalize(self):
        idx = list( set(range(len(self.model.odes))) - set(self.solve_pruned.keys()) )
        eqs = { i:self.model.odes[i] for i in idx }

        for l in eqs.keys(): #Substitutes the values of the algebraic system
            for k in self.solved_pruned.keys(): eqs[l]=eqs[l].subs(sympy.Symbol('s%d' % k), self.solved_pruned[k])

        for i in eqs.keys():
            for par in self.model.parameters: eqs[i] = sympy.simplify(eqs[i].subs(par.name, par.value))

        self.eqs_for_tropicalization = eqs

        return eqs
    
    def final_tropicalization(self):
        tropicalized = {}
        
        for j in sorted(self.eqs_for_tropicalization.keys()):
            if type(self.eqs_for_tropicalization[j]) == sympy.Mul: print sympy.solve(sympy.log(j), dict = True) #If Mul=True there is only one monomial
            elif self.eqs_for_tropicalization[j] == 0: print 'there are no monomials'
            else:            
                ar = self.eqs_for_tropicalization[j].args #List of the terms of each equation  
                asd=0 
                for l, k in enumerate(ar):
                    p = k
                    for f, h in enumerate(ar):
                       if k != h:
                          p *= sympy.Heaviside(sympy.log(abs(k)) - sympy.log(abs(h)))
                    asd +=p
                tropicalized[j] = asd

        self.tropical_eqs = tropicalized

        return tropicalized

def tropicalization(eqs_for_tropicalization):
    tropicalized = {}

    for j in sorted(eqs_for_tropicalization.keys()):
        if type(eqs_for_tropicalization[j]) == sympy.Mul: print sympy.solve(sympy.log(j), dict = True) #If Mul=True there is only one monomial
        elif eqs_for_tropicalization[j] == 0: print 'there are not monomials'
        else:            
            ar = eqs_for_tropicalization[j].args #List of the terms of each equation  
            asd=0 
            for l, k in enumerate(ar):
                p = k
                for f, h in enumerate(ar):
                   if k != h:
                      p *= sympy.Heaviside(sympy.log(abs(k)) - sympy.log(abs(h)))
                asd +=p
            tropicalized[j] = asd
    return tropicalized

from pysb.examples.tyson_oscillator import model
tro = Tropical(model)
tro.tropicalize()
