# import the pysb module and all its methods and functions
from pysb import *

# instantiate a model
Model()

# declare monomers
Monomer('C8', ['b'])
Monomer('Bid', ['b', 'S'])

# input the parameter values
Parameter('kf', 1.04e-06)
Parameter('kr', 1.04e-06)
Parameter('kc', 1.04e-06)

# now input the rules
Rule('C8_Bid_bind', C8(b=None) + Bid(b=None, S=None) <>
                       C8(b=1) % Bid(b=1, S=None), *[kf, kr]) 


