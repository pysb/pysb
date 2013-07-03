import networkx
import sympy
import re
import copy
import numpy
import sympy.parsing.sympy_parser
from pysb.integrate import odesolve
from collections    import Mapping


class Tropical:
    def __init__(self):
        self.model            = None
        self.t                = None # timerange used
        self.y                = None # odes solution, numpy array
        self.imposed_distance = None
        self.slaves           = None
        self.graph            = None
        self.cycles           = None
        self.conservation     = None
        self.full_names       = {}

    def __repr__(self):
        return "<%s '%s' (slaves: %s, cycles: %d) at 0x%x>" % \
            (self.__class__.__name__, self.model.name,
             self.slaves.__repr__(),
             len(self.cycles),
             id(self))

# Constructor function
def tropicalize(model, t, ignore=1, epsilon=0.1, rho=1, verbose=False):
    tropical = Tropical()
    tropical.model = model
    if verbose: print "Solving Simulation"
    tropical.y = odesolve(model, t)
    
    # Only concrete species are considered, and the names must be made to match
    names               = [n for n in filter(lambda n: n.startswith('__'), tropical.y.dtype.names)]
    tropical.y          = tropical.y[names]
    names               = [n.replace('__','') for n in names]
    tropical.y.dtype    = [(n,'<f8') for n in names]
    
    if verbose: print "Computing Imposed Distances"
    # Not saving imposed distance, since it's shortcutting
    dist = imposed_distance(model, tropical.y[ignore:], verbose, epsilon)
    tropical.slaves = slaves(dist, epsilon)
    if verbose: print "Constructing Graph"
    tropical.graph = construct_graph(model)
    if verbose: print "Computing Cycles"
    tropical.cycles = [c[0:(len(c)-1)]  for c in networkx.simple_cycles(tropical.graph) ]
    if verbose: print "Computing Conservation laws"
    (tropical.conservation, conserved_variabled) = mass_conserved(model, tropical.cycles)
    return tropical

# Helper class to use evalf with a ndarray
class FilterNdarray(Mapping):
    def __init__(self, source):
        self.source = source
        self.t = 0

    def __getitem__(self, key): return self.source[key.name][self.t]

    def __len__(self): return len(self.source)

    def __iter__(self):
        for key in self.source:
            yield key

    def set_time(self, t):
        self.t = t
        return self

# Compute imposed distances of a model
def imposed_distance(model, y, verbose=False, epsilon=None):
    distances = []

    symy = FilterNdarray(y) # Create a filtered view of the ode solution, suitable for sympy

    # Loop through all equations (i is equation number)
    for i, eq in enumerate(model.odes):
        eq        = eq.subs('s%d' % i, 's%dstar' % i)
        sol       = sympy.solve(eq, sympy.Symbol('s%dstar' % i)) # Find equation of imposed trace
        max       = None # maximum for a single solution
        min       = None # Minimum between all possible solution

        # The minimum of the maximum of all possible solutions
        # Note: It should prefer real solutions over complex, but this isn't coded to do that 
        #       and the current test case is all complex
        for j in range(len(sol)):  # j is solution j for equation i (mostly likely never greater than 2)
            for p in model.parameters: sol[j] = sol[j].subs(p.name, p.value) # Substitute parameters
            trace = y['s%d' % i]
            for t in range(y.size):
                current = abs(sol[j].evalf(subs=symy.set_time(t)) - trace[t])
                if max == None or current > max:
                    max = current
                if epsilon != None and current > epsilon:
                    break
            if j==0:
                min = max
            else:
                if max < min: min = max
        distances.append(min)
        if verbose: print "  * Equation",i," distance =",max
    return distances

def slaves(distances, epsilon):
    slaves = []
    for i,d in enumerate(distances):
        if (d != None and d < epsilon): slaves.append(i)
    return slaves

# This is a function which builds the edges according to the nodes
def r_link(graph, s, r, **attrs):
    nodes = (s, r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = reversed(nodes)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)

def construct_graph(model):
    if(model.odes == None or model.odes == []):
        pysb.bng.generate_equations(model)

    graph = networkx.DiGraph(rankdir="LR")
    ic_species = [cp for cp, parameter in model.initial_conditions]
    for i, cp in enumerate(model.species):
        species_node = i
        graph.add_node(species_node,
                       label=species_node)
    for i, reaction in enumerate(model.reactions):       
        reactants = set(reaction['reactants'])
        products = set(reaction['products']) 
        attr_reversible = {}
        for s in reactants:
            for p in products:
                r_link(graph, s, p, **attr_reversible)
    return graph

#This function finds conservation laws from the conserved cycles
def mass_conserved(model, cycles, verbose=False):
    if(model.odes == None or model.odes == []):
        pysb.bng.generate_equations(model)
    h = [] # Array to hold conservation equation
    g = [] # Array to hold corresponding lists of free variables in conservation equations
    for i, item in enumerate(cycles):
        b = 0
        u = 0
        for j, specie in enumerate(item):
            b += model.odes[cycles[i][j]]
        if b == 0:
            g.append(item)
            for l,k in enumerate(item):
                u += sympy.Symbol('s%d' % cycles[i][l])    
            h.append(u-sympy.Symbol('C%d'%i))
            if verbose: print '  cycle%d'%i, 'is conserved'
    return h, g

def slave_equations(model, slaves):
    if(model.odes == None or model.odes == []):
        eq = model.odes
    slave_conserved_eqs = []
    for i, j in enumerate(slaves):
        slave_conserved_eqs.append(model.odes[slaves[i]])
    return slave_conserved_eqs

def find_nearest_zero(array):
    idx = (numpy.abs(array)).argmin()
    return array[idx]

# Make sure this is the "ignore:" y
def pruned_equations(model, y, conservation_laws, slaves, rho=1):
    pruned_eqs = slave_equations(model, slaves)
    eqs        = copy.deepcopy(pruned_eqs)
     # Create a filtered view of the ode solution, suitable for sympy

    for i, eq in enumerate(eqs):
        ble = eq.as_coefficients_dict().keys() # Get monomials
        for l, m in enumerate(ble): #Compares the monomials to find the pruned system
            m_ready = m # Monomial to compute with
            m_elim  = m # Monomial to save
            for p in model.parameters: m_ready = m_ready.subs(p.name, p.value) # Substitute parameters
            for k in range(len(ble)):
                if (k+l+1) <= (len(ble)-1):
                    ble_ready = ble[k+l+1] # Monomial to compute with
                    ble_elim  = ble[k+l+1] # Monomial to save
                    for p in model.parameters: ble_ready = ble_ready.subs(p.name, p.value) # Substitute parameters
                    diff = [m_ready.evalf(subs=symy.set_time(t)) - ble_ready.evalf(subs=symy.set_time(t)) for t in range(y.size)]
                    diff = find_nearest_zero(diff)
                    if diff > 0 and abs(diff) > rho:
                       pruned_eqs[i] = pruned_eqs[i].subs(ble_elim, 0)
                    if diff < 0 and abs(diff) > rho:
                       pruned_eqs[i] = pruned_eqs[i].subs(m_elim, 0)
    for law in conservation_laws: #Add the conservation laws to the pruned system
        pruned_eqs.append(law)
    return pruned_eqs

