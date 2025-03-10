"""A trivial example of time dependent rates"""

from pysb import Monomer, Parameter, Rule, Model, Expression, time
from sympy import exp

Model()

Monomer('A')

Parameter('kA_syn', 1e0)

# exponential decay of kA_syn
Expression('kA_syn_time', kA_syn * exp(-time))

Rule('synthesize_A', None >> A(), kA_syn)