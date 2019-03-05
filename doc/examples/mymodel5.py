# import the pysb module and all its methods and functions
from pysb import *


def catalyze(enz, sub, site, state1, state2, kf, kr, kc):   # function call
    """2-step catalytic process"""                          # reaction name
    r1_name = '%s_assoc_%s' % (enz.name, sub.name)          # name of association reaction for rule
    r2_name = '%s_diss_%s' % (enz.name, sub.name)           # name of dissociation reaction for rule
    E = enz(b=None)                                         # define enzyme state in function
    S = sub({'b': None, site: state1})                      # define substrate state in function
    ES = enz(b=1) % sub({'b': 1, site: state1})             # define state of enzyme:substrate complex
    P = sub({'b': None, site: state2})                      # define state of product
    Rule(r1_name, E + S | ES, kf, kr)                       # rule for enzyme + substrate association (bidirectional)
    Rule(r2_name, ES >> E + P, kc)                          # rule for enzyme:substrate dissociation  (unidirectional)
   
# instantiate a model
Model()

# declare monomers
Monomer('C8', ['b'])
Monomer('Bid', ['b', 'S'], {'S':['u', 't']})

# input the parameter values
Parameter('kf', 1.0e-07)
Parameter('kr', 1.0e-03)
Parameter('kc', 1.0)

# OLD RULES
# Rule('C8_Bid_bind', C8(b=None) + Bid(b=None, S='u') | C8(b=1) % Bid(b=1, S='u'), *[kf, kr])
# Rule('tBid_from_C8Bid', C8(b=1) % Bid(b=1, S='u') >> C8(b=None) + Bid(b=None, S='t'), kc)
#
# NEW RULES
# Catalysis
catalyze(C8, Bid, 'S', 'u', 't', kf, kr, kc)


# initial conditions
Parameter('C8_0', 1000)
Parameter('Bid_0', 10000)
Initial(C8(b=None), C8_0)
Initial(Bid(b=None, S='u'), Bid_0)

# Observables
Observable('obsC8', C8(b=None))
Observable('obsBid', Bid(b=None, S='u'))
Observable('obstBid', Bid(b=None, S='t'))

