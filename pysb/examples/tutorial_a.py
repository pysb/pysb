from pysb import *

Model()
Monomer('A')
Parameter('k', 3.0)
Rule('synthesize_A', None >> A(), k)
