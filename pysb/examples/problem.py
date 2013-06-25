from numpy import *
from sympy import *

x = array([(1.1, 2.2), (3.3, 4.4), (5.5, 6.6)], dtype=[('s0', '<f8'), ('s1', '<f8')])


        

x = {Symbol('s0'):[1.1, 3.3, 5.5], Symbol('s1'):[2.2, 4.4, 6.6]}

eq = 2.0 * Symbol('s0') - Symbol('s1')

eq.evalf(subs=x)

from collections import Mapping
class FilterNdarray(Mapping):
    def __init__(self, source, t):
        self.source = source
        self.t = t

    def __getitem__(self, key):
        return self.source[key.name][self.t]

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        for key in self.source:
            yield key
            
y = FilterNdarray(x, 0)

# Monkey patch?
def symStringMap(self, key):
    return self[key.name]
    
x.__getitem__ = symStringMap


# Problem in sympy with time series
from sympy import *

x = {Symbol('s0'):[1.1, 3.3, 5.5], Symbol('s1'):[2.2, 4.4, 6.6]}

eq = 2.0 * Symbol('s0') - Symbol('s1')

eq.evalf(subs=x)


# Simplification
from sympy import *

x = {Symbol('s0'):[1.1, 3.3, 5.5], Symbol('s1'):[2.2, 4.4, 6.6]}

eq = 2.0*Symbol('s1')

eq.evalf(subs=x)