from numpy import *
from sympy import *

x = array([(1.1, 2.2), (3.3, 4.4), (5.5, 6.6)], dtype=[('s0', '<f8'), ('s1', '<f8')])

eq = 2.0 * Symbol('s0') - sqrt(Symbol('s1'))

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

    def set_time(self, t):
        self.t = t
        return self

y = FilterNdarray(x, 0)
[eq.evalf(subs=y.set_time(t)) for t in range(x.size)]
