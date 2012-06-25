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


