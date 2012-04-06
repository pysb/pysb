from pysb import *

Model()

Monomer('Kinase', ['b'])
Monomer('Substrate', ['b', 'y'], {'y': ['U','P']})


def catalyze(enz, sub, site, state1, state2, kf, kr, kc):
    """2-step catalytic process"""
    r1_name = '%s_bind_%s' % (enz.name, sub.name)
    r2_name = '%s_produce_%s' % (enz.name, sub.name)
    E = enz(b=None)
    S = sub({'b': None, site: state1})
    ES = enz(b=1) % sub({'b': 1, site: state1})
    P = sub({'b': None, site: state2})
    Rule(r1_name, E + S <> ES, kf, kr)
    Rule(r2_name, ES >> E + P, kc)


Parameter('kf', 1)
Parameter('kr', 1)
Parameter('kc', 1)
catalyze(Kinase, Substrate, 'y', 'U', 'P', kf, kr, kc)

Parameter('dummy_0', 1)
Initial(Kinase(b=None), dummy_0)
Initial(Substrate(b=None,y='U'), dummy_0)
