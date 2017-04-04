from pysb import *
from pysb.macros import catalyze_state

Model()

Parameter('vol', 10.) # volume (arbitrary units)

Parameter('kf',   1./vol.value)
Parameter('kr',   1000)
Parameter('kcat', 100)

Monomer('E', ['s'])
Monomer('S', ['e', 'state'], {'state': ['0','1']})

catalyze_state(E(), 's', S(), 'e', 'state', '0', '1', [kf,kr,kcat])

Observable("E_free",     E(s=None))
Observable("S_free",     S(e=None, state='0'))
Observable("ES_complex", E(s=1) % S(e=1))
Observable("Product",    S(e=None, state='1'))

Parameter("Etot", 1.*vol.value)
Initial(E(s=None), Etot)

Parameter('S0', 10.*vol.value)
Initial(S(e=None, state='0'), S0)
