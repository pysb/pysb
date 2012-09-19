# Adapted from:
# Input-output behavior of ErbB signaling pathways as revealed by a mass action
# model trained against dynamic data. William W Chen, Birgit Schoeberl, Paul J
# Jasper, Mario Niepel, Ulrik B Nielsen, Douglas A Lauffenburger & Peter K
# Sorger. doi:10.1038/msb.2008.74

from pysb import *
from pysb.macros import catalyze
from pysb.integrate import odesolve
from pylab import linspace, plot, legend, show

Model()

Monomer('Ras', ['k'])
Monomer('Raf', ['s', 'k'], {'s': ['u', 'p']})
Monomer('MEK', ['s218', 's222', 'k'], {'s218': ['u', 'p'], 's222': ['u', 'p']})
Monomer('ERK', ['t185', 'y187'], {'t185': ['u', 'p'], 'y187': ['u', 'p']})
Monomer('PP2A', ['ppt'])
Monomer('MKP', ['ppt'])

kf_bind = 1e-5
kr_bind = 1e-1
kcat_phos = 1e-1
kcat_dephos = 3e-3
klist_bind = [kf_bind, kr_bind]
klist_phos = klist_bind + [kcat_phos]
klist_dephos = klist_bind + [kcat_dephos]

catalyze(Ras, 'k', Raf(s='u'), 's', Raf(s='p'), klist_phos)
catalyze(PP2A, 'ppt', Raf(s='p', k=None), 's', Raf(s='u', k=None), klist_dephos)

catalyze(Raf(s='p'), 'k', MEK(s218='u'), 's218', MEK(s218='p'), klist_phos)
catalyze(Raf(s='p'), 'k', MEK(s218='p', s222='u'), 's222', MEK(s218='p', s222='p'), klist_phos)
catalyze(PP2A, 'ppt', MEK(s218='p', s222='u', k=None), 's218', MEK(s218='u', s222='u', k=None), klist_dephos)
catalyze(PP2A, 'ppt', MEK(s218='p', s222='p', k=None), 's222', MEK(s218='p', s222='u', k=None), klist_dephos)

catalyze(MEK(s218='p', s222='p'), 'k', ERK(t185='u'), 't185', ERK(t185='p'), klist_phos)
catalyze(MEK(s218='p', s222='p'), 'k', ERK(t185='p', y187='u'), 'y187', ERK(t185='p', y187='p'), klist_phos)
catalyze(MKP, 'ppt', ERK(t185='p', y187='u'), 't185', ERK(t185='u', y187='u'), klist_dephos)
catalyze(MKP, 'ppt', ERK(t185='p', y187='p'), 'y187', ERK(t185='p', y187='u'), klist_dephos)

Initial(Ras(k=None), Parameter('Ras_0', 6e4))
Initial(Raf(s='u', k=None), Parameter('Raf_0', 7e4))
Initial(MEK(s218='u', s222='u', k=None), Parameter('MEK_0', 3e6))
Initial(ERK(t185='u', y187='u'), Parameter('ERK_0', 7e5))
Initial(PP2A(ppt=None), Parameter('PP2A_0', 2e5))
Initial(MKP(ppt=None), Parameter('MKP_0', 1.7e4))

Observable('MEK_pp', MEK(s218='p', s222='p'))
Observable('ERK_pp', ERK(t185='p', y187='p'))

if __name__ == '__main__':
    tspan = linspace(0, 1200)
    yfull = odesolve(model, tspan)
    plot(tspan, yfull['MEK_pp'])
    plot(tspan, yfull['ERK_pp'])
    legend(('pMEK', 'pERK'), loc='upper left')
    show()
