"""Elements for Figure 2C from the PySB publication"""

from __future__ import print_function
from pysb import *
from pysb.macros import bind_table
from pysb.bng import generate_network, generate_equations

Model()

Monomer('Bid', ['bf'])
Monomer('Bim', ['bf'])
Monomer('Bad', ['bf'])
Monomer('Bik', ['bf'])
Monomer('Noxa', ['bf'])
Monomer('Hrk', ['bf'])
Monomer('Puma', ['bf'])
Monomer('Bmf', ['bf'])

Monomer('Bcl2', ['bf'])
Monomer('BclXL', ['bf'])
Monomer('BclW', ['bf'])
Monomer('Mcl1', ['bf'])
Monomer('Bfl1', ['bf'])

Initial(Bid(bf=None), Parameter('Bid_0', 1))
Initial(Bim(bf=None), Parameter('Bim_0', 1))
Initial(Bad(bf=None), Parameter('Bad_0', 1))
Initial(Bik(bf=None), Parameter('Bik_0', 1))
Initial(Noxa(bf=None), Parameter('Noxa_0', 1))
Initial(Hrk(bf=None), Parameter('Hrk_0', 1))
Initial(Puma(bf=None), Parameter('Puma_0', 1))
Initial(Bmf(bf=None), Parameter('Bmf_0', 1))

Initial(Bcl2(bf=None), Parameter('Bcl2_0', 1))
Initial(BclXL(bf=None), Parameter('BclXL_0', 1))
Initial(BclW(bf=None), Parameter('BclW_0', 1))
Initial(Mcl1(bf=None), Parameter('Mcl1_0', 1))
Initial(Bfl1(bf=None), Parameter('Bfl1_0', 1))

# This is the code shown for "Example Macro Call" (not printed here)
bind_table([[       Bcl2,  BclXL,  BclW,  Mcl1,  Bfl1],
            [Bid,     66,     12,    10,    10,    53],
            [Bim,     10,     10,    38,    10,    73],
            [Bad,     11,     10,    60,  None,  None],
            [Bik,    151,     10,    17,   109,  None],
            [Noxa,  None,   None,  None,    19,  None],
            [Hrk,   None,     92,  None,  None,  None],
            [Puma,    18,     10,    25,    10,    59],
            [Bmf,     24,     10,    11,    23,  None]],
            'bf', 'bf', kf=1e3)

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
    assert num_rules == 28, "number of rules not as expected"
    assert num_odes == 41, "number of odes not as expected"
