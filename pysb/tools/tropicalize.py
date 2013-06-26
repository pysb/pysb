import sympy
#from sympy.functions.elementary.complexes import Abs
from pysb.integrate import odesolve
from collections    import Mapping

# NOTES TO RESTART
# (a) Is there a better name for a species than s0, s1, ...?
# (b) How to organize this as a class? Co-monadic structure seems overkill, when most pieces are functions
#     ? Maybe functions inside Tropical class, or just don't export them from module
# (c) Add test cases for given interface
# (d) git rebase all this to a "tropical" branch
# (e) will zeros speed it up?

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
def tropicalize(model, t, ignore, epsilon=1e-6, rho=1, verbose=False):
    tropical = Tropical()
    tropical.model = model
    if verbose: print "Computing Imposed Distances"
    tropical.imposed_distance = imposed_distance(model, t, ignore, verbose)
    tropical.slaves = slaves(tropical.imposed_distance, epsilon)
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
def imposed_distance(model, t, ignore, verbose=False):
    distances = []

    # Find ode solution via specified parameters
    x = odesolve(tyson, t)

    # Ignore first couple points, till system reaches quasi-equilibrium
    x = x[ignore:]

    # Only concrete species are considered, and the names must be made to match
    names      = [n for n in filter(lambda n: n.startswith('__'), x.dtype.names)]
    x          = x[names]
    names      = [n.replace('__','') for n in names]
    x.dtype    = [(n,'<f8') for n in names]
    symx       = FilterNdarray(x) # Create a filtered view of the ode solution, suitable for sympy
    trabajando = [0.0]*x.size

    # Loop through all equations (i is equation number)
    for i, eq in enumerate(model.odes):
        eq        = eq.subs('s%d' % i, 's%dstar' % i)
        sol       = sympy.solve(eq, sympy.Symbol('s%dstar' % i)) # Find equation of imposed trace
        max       = -1 # Start with no distance between imposed trace and computed trace for this species

        for j in range(len(sol)):  # j is solution j for equation i (mostly likely never greater than 2)
            for p in model.parameters: sol[j] = sol[j].subs(p.name, p.value) # Substitute parameters
            #for t in range(x.size): trabajando[t] = sol[j].evalf(subs=symx.set_time(t))
            #prueba = abs(trabajando - x['s%d' % i]).max()
            prueba = abs([sol[j].evalf(subs=symx.set_time(t)) for t in range(x.size)] - x['s%d' % i]).max()
            if(prueba > max): max = prueba
        if (max < 0): distances.append(None)
        else:         distances.append(max)
        if verbose: print "  * Equation",i," distance =",max
    return distances

def slaves(distances, epsilon):
    slaves = []
    for i,d in enumerate(distances):
        if (d != None and d < epsilon): slaves.append('s%d'%i)
    return slaves