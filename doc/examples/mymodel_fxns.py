# import the pysb module and all its methods and functions
from pysb import *
from pysb.macros import *

# some functions to make life easy
site_name = 'b'
def catalyze_b(enz, sub, product, klist):
    """Alias for pysb.macros.catalyze with default binding site 'b'.
    """
    return catalyze(enz, site_name, sub, site_name, product, klist)

def bind_table_b(table):
    """Alias for pysb.macros.bind_table with default binding sites 'bf'.
    """
    return bind_table(table, site_name, site_name)

# Default forward, reverse, and catalytic rates
KF = 1e-6
KR = 1e-3
KC = 1

# Bid activation rates
bid_rates = [        1e-7, 1e-3, 1] #

# Bcl2 Inhibition Rates
bcl2_rates = [1.428571e-05, 1e-3] # 1.0e-6/v_mito

# instantiate a model
Model()

# declare monomers
Monomer('C8',    ['b'])
Monomer('Bid',   ['b', 'S'], {'S':['u', 't', 'm']})
Monomer('Bax',   ['b', 'S'], {'S':['i', 'a', 'm']})
Monomer('Bak',   ['b', 'S'], {'S':['i', 'a']})
Monomer('BclxL', ['b', 'S'], {'S':['c', 'm']})
Monomer('Bcl2', ['b'])
Monomer('Mcl1', ['b'])

# Activate Bid
catalyze_b(C8, Bid(S='u'), Bid(S='t'), [KF, KR, KC])

# Activate Bax/Bak
catalyze_b(Bid(S='m'), Bax(S='i'), Bax(S='m'), bid_rates)
catalyze_b(Bid(S='m'), Bak(S='i'), Bak(S='a'), bid_rates)

# Bid, Bax, BclxL "transport" to the membrane
equilibrate(Bid(b=None, S='t'),   Bid(b=None, S='m'), [1e-1, 1e-3])
equilibrate(Bax(b=None, S='m'),   Bax(b=None, S='a'), [1e-1, 1e-3])
equilibrate(BclxL(b=None, S='c'), BclxL(b=None, S='m'), [1e-1, 1e-3])


bind_table_b([[                  Bcl2,  BclxL(S='m'),       Mcl1],
              [Bid(S='m'), bcl2_rates,  bcl2_rates,   bcl2_rates],
              [Bax(S='a'), bcl2_rates,  bcl2_rates,         None],
              [Bak(S='a'),       None,  bcl2_rates,   bcl2_rates]])

# initial conditions
Parameter('C8_0',    1e4)
Parameter('Bid_0',   1e4)
Parameter('Bax_0',  .8e5)
Parameter('Bak_0',  .2e5)
Parameter('BclxL_0', 1e3)
Parameter('Bcl2_0',  1e3)
Parameter('Mcl1_0',  1e3)

Initial(C8(b=None), C8_0)
Initial(Bid(b=None, S='u'), Bid_0)
Initial(Bax(b=None, S='i'), Bax_0)
Initial(Bak(b=None, S='i'), Bak_0)
Initial(BclxL(b=None, S='c'), BclxL_0)
Initial(Bcl2(b=None), Bcl2_0)
Initial(Mcl1(b=None), Mcl1_0)

# Observables
Observable('obstBid', Bid(b=None, S='m'))
Observable('obsBax', Bax(b=None, S='a'))
Observable('obsBak', Bax(b=None, S='a'))
Observable('obsBaxBclxL', Bax(b=1, S='a')%BclxL(b=1, S='m'))
