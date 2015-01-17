"""Elements for Figure 2B from the PySB publication"""

from __future__ import print_function
from pysb import *
from pysb.macros import *
from pysb.bng import generate_equations

Model()

Monomer('Bax', ['s1', 's2'])
Initial(Bax(s1=None, s2=None), Parameter('Bax_0', 1))

ktable = [[1, 1]] * 5

# This is the code shown for "Example Macro Call" (not printed here)
assemble_pore_sequential(Bax, 's1', 's2', 6, ktable)

generate_equations(model)

num_rules = len(model.rules)
num_odes = len(model.odes)

print("BNGL Rules")
print("==========")
print(num_rules, "rules")
print()
print("ODEs")
print("====")
print(num_odes, "ODEs")

def test_fig2c():
    assert num_rules == 5, "number of rules not as expected"
    assert num_odes == 6, "number of odes not as expected"
