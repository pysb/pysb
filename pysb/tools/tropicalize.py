import networkx
import sympy
import re
import copy
import numpy
import sympy.parsing.sympy_parser
from pysb.integrate import odesolve
from collections    import Mapping

# Helper class to use evalf with a ndarray
class FilterNdarray(Mapping):
    def __init__(self, source):
        self.source = source
        self.t = 0

    def __getitem__(self, key):
        # WARNING: This is a monkey patch when evalf sends a number to getitem instead of a symbol
        try:
            return self.source[key.name][self.t]
        except:
            return key.name

    def __len__(self): return len(self.source)

    def __iter__(self):
        for key in self.source:
            yield key

    def set_time(self, t):
        self.t = t
        return self

class Tropical:
    def __init__(self, model):
        self.model            = model
        self.t                = None # timerange used
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

    def tropicalize(self, t, ignore=1, epsilon=0.1, rho=1, verbose=False):
        if verbose: print "Solving Simulation"
        self.y = odesolve(self.model, t)
    
        # Only concrete species are considered, and the names must be made to match
        names           = [n for n in filter(lambda n: n.startswith('__'), self.y.dtype.names)]
        self.y          = self.y[names]
        names           = [n.replace('__','') for n in names]
        self.y.dtype    = [(n,'<f8') for n in names]
    
        if verbose: print "Computing Imposed Distances"
        # Not saving imposed distance, since it's shortcutting
        dist = self.imposed_distance(self.y[ignore:], verbose, epsilon)
        self.find_slaves(dist, epsilon)
        if verbose: print "Constructing Graph"
        self.construct_graph()
        if verbose: print "Computing Cycles"
        self.cycles = [c[0:(len(c)-1)]  for c in networkx.simple_cycles(self.graph) ]
        if verbose: print "Computing Conservation laws"
        (self.conservation, self.conserve_var) = self.mass_conserved()
        if verbose: print "Pruning Equations"
        self.pruned = self.pruned_equations(self.y[ignore:], rho)

# FIXME: THIS STEP IS BROKEN DUE TO THE ADDITION OF CYCLES
        #if verbose: print "Solving pruned equations"
        #self.sol_pruned = self.solve_pruned()
    
        return self

    # Compute imposed distances of a model
    def imposed_distance(self, y, verbose=False, epsilon=None):
        distances = []

        symy = FilterNdarray(y) # Create a filtered view of the ode solution, suitable for sympy

        # Loop through all equations (i is equation number)
        for i, eq in enumerate(self.model.odes):
            eq        = eq.subs('s%d' % i, 's%dstar' % i)
            sol       = sympy.solve(eq, sympy.Symbol('s%dstar' % i)) # Find equation of imposed trace
            max       = None # maximum for a single solution
            min       = None # Minimum between all possible solution

            # The minimum of the maximum of all possible solutions
            # Note: It should prefer real solutions over complex, but this isn't coded to do that 
            #       and the current test case is all complex
            for j in range(len(sol)):  # j is solution j for equation i (mostly likely never greater than 2)
                for p in self.model.parameters: sol[j] = sol[j].subs(p.name, p.value) # Substitute parameters
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

    def find_slaves(self, distances, epsilon):
        self.slaves = []
        for i,d in enumerate(distances):
            if (d != None and d < epsilon): self.slaves.append(i)
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
        slave_conserved_eqs = []
        for i, j in enumerate(self.slaves):
            slave_conserved_eqs.append(self.model.odes[self.slaves[i]])
        return slave_conserved_eqs

    def find_nearest_zero(self, array):
        idx = (numpy.abs(array)).argmin()
        return array[idx]

    # Make sure this is the "ignore:" y
    def pruned_equations(self, y, rho=1):
        pruned_eqs = self.slave_equations()
        eqs        = copy.deepcopy(pruned_eqs)
        symy       = FilterNdarray (y)

        for i, eq in enumerate(eqs):
            ble = eq.as_coefficients_dict().keys() # Get monomials
            for l, m in enumerate(ble): #Compares the monomials to find the pruned system
                m_ready = m # Monomial to compute with
                m_elim  = m # Monomial to save
                for p in self.model.parameters: m_ready = m_ready.subs(p.name, p.value) # Substitute parameters
                for k in range(len(ble)):
                    if (k+l+1) <= (len(ble)-1):
                        ble_ready = ble[k+l+1] # Monomial to compute with
                        ble_elim  = ble[k+l+1] # Monomial to save
                        for p in self.model.parameters: ble_ready = ble_ready.subs(p.name, p.value) # Substitute parameters
                        diff = [m_ready.evalf(subs=symy.set_time(t)) - ble_ready.evalf(subs=symy.set_time(t)) for t in range(y.size)]
                        diff = self.find_nearest_zero(diff)
                        #print i, eq, l, m, k, diff
                        if diff > 0 and abs(diff) > rho:
                           pruned_eqs[i] = pruned_eqs[i].subs(ble_elim, 0)
                        if diff < 0 and abs(diff) > rho:
                           pruned_eqs[i] = pruned_eqs[i].subs(m_elim, 0)
        for law in self.conservation: #Add the conservation laws to the pruned system
            pruned_eqs.append(law)
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
