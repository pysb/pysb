import sympy
#from sympy.functions.elementary.complexes import Abs
from pysb.integrate import odesolve
from collections    import Mapping

class Tropical:
    def __init__(self):
        self.model            = None
        self.imposed_distance = None
        self.slaves           = None
        self.full_names       = {}

    def __repr__(self):
        return "<%s '%s' (slaves: %s) at 0x%x>" % \
            (self.__class__.__name__, self.model.name,
             self.slaves.__repr__(),
             id(self))

# Constructor function
def tropicalize(model, t, ignore=1, epsilon=0.1, rho=1, verbose=False):
    tropical = Tropical()
    tropical.model = model
    if verbose: print "Computing Imposed Distances"
    # Not saving imposed distance, since it's shortcutting
    dist = imposed_distance(model, t, ignore, verbose, epsilon)
    tropical.slaves = slaves(dist, epsilon)
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
def imposed_distance(model, t, ignore=1, verbose=False, epsilon=None):
    distances = []

    # Find ode solution via specified parameters
    x = odesolve(model, t)

    # Ignore first couple points, till system reaches quasi-equilibrium
    x = x[ignore:]

    # Only concrete species are considered, and the names must be made to match
    names      = [n for n in filter(lambda n: n.startswith('__'), x.dtype.names)]
    x          = x[names]
    names      = [n.replace('__','') for n in names]
    x.dtype    = [(n,'<f8') for n in names]
    symx       = FilterNdarray(x) # Create a filtered view of the ode solution, suitable for sympy

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
            trace = x['s%d' % i]
            for t in range(x.size):
                current = abs(sol[j].evalf(subs=symx.set_time(t)) - trace[t])
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
        if (d != None and d < epsilon): slaves.append('s%d'%i)
    return slaves