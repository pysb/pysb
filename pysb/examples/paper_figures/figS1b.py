"""Elements for Figure S1B from the PySB publication"""

from __future__ import print_function
from pysb import *
from pysb.macros import catalyze_one_step
from pysb.bng import generate_network, generate_equations
import re

Model()

Monomer('C8', ['bf'])
Monomer('Bid', ['bf', 'state'], {'state': ['U', 'T']})

kf = Parameter('kf', 1)

Initial(C8(bf=None), Parameter('C8_0', 100))
Initial(Bid(bf=None, state='U'), Parameter('Bid_0', 100))

# This is the code shown for "Example Macro Call" (not printed here)
catalyze_one_step(C8(bf=None), Bid(state='U', bf=None),
                  Bid(state='T', bf=None), kf)

bng_code = generate_network(model)
# Merge continued lines
bng_code = bng_code.replace('\\\n', '')
generate_equations(model)

num_rules = len(model.rules)
num_odes = len(model.odes)

print("BNGL Rules")
print("==========")
for line in bng_code.split("\n"):
    for rule in model.rules:
        match = re.match(r'^\s*%s:\s*(.*)' % rule.name, line)
        if match:
            print(match.group(1))
print()
print("ODEs")
print("====")
for species, ode in zip(model.species, model.odes):
    print("%s: %s" % (species, ode))

def test_figs1b():
    assert num_rules == 1, "number of rules not as expected"
    assert num_odes == 3, "number of odes not as expected"
