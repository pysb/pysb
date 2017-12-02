"""Elements for Figure 2A from the PySB publication"""

from __future__ import print_function
from pysb import *
from pysb.bng import generate_network, generate_equations
import re


# This code (pygmentized) is shown in figure S1A as "Basic Implementation"
def catalyze(enz, e_site, sub, s_site, prod, klist):
    kf, kr, kc = klist   # Get the parameters from the list

    # Create the rules
    rb = Rule('bind_%s_%s' % (enz().monomer.name, sub().monomer.name),
           enz({e_site:None}) + sub({s_site:None}) |
           enz({e_site:1}) % sub({s_site:1}),
           kf, kr)
    rc = Rule('catalyze_%s%s_to_%s' %
           (enz().monomer.name, sub().monomer.name, prod().monomer.name),
           enz({e_site:1}) % sub({s_site:1}) >>
           enz({e_site:None}) + prod({s_site:None}),
           kc)
    return [rb, rc]

Model()

Monomer('C8', ['bf'])
Monomer('Bid', ['bf', 'state'], {'state': ['U', 'T']})

klist = [Parameter('kf', 1), Parameter('kr', 1), Parameter('kc', 1)]

Initial(C8(bf=None), Parameter('C8_0', 100))
Initial(Bid(bf=None, state='U'), Parameter('Bid_0', 100))

# This is the code shown for "Example Macro Call" (not printed here)
catalyze(C8, 'bf', Bid(state='U'), 'bf', Bid(state='T'), klist)

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

def test_fig2a():
    assert num_rules == 2, "number of rules not as expected"
    assert num_odes == 4, "number of odes not as expected"
