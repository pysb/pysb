from pysb import *

Model()

Monomer('Kinase', ['b'])
Monomer('Inhibitor', ['b'])


def inhibit(targ, inh, kf, kr):
    """inhibition by complexation/sequestration"""
    r_name = '%s_inhibited_by_%s' % (targ.name, inh.name)
    T = targ(b=None)
    I = inh(b=None)
    TI = targ(b=1) % inh(b=1)
    Rule(r_name, T + I <> TI, kf, kr)


Parameter('kf', 1)
Parameter('kr', 1)
inhibit(Kinase, Inhibitor, kf, kr)

Parameter('dummy_0', 1)
Initial(Kinase(b=None), dummy_0)
Initial(Inhibitor(b=None), dummy_0)
