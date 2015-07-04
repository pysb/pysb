from pysb import *
from pysb.macros import *
Model()



# Parameter('kf', 2)
# Parameter('kr', 2)

Monomer('E', ['b'])
Monomer('S', ['b'])
Monomer('P')
catalyze(E(), 'b', S(), 'b', P(), (1e-4, 1e-1, 1))

Parameter('E_0',100000)
Parameter('S_0', 50000)
Initial(E(b=None), E_0)
Initial(S(b=None), S_0)

