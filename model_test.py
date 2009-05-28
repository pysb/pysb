from Pysb import *

Model('test')

Compartment('extracellular', dimension=3)
Compartment('cytoplasm', dimension=3)
Compartment('membrane', dimension=2, neighbors=[extracellular, cytoplasm])

Monomer('egf', 'R')
Monomer('egfr', ['L', 'D', 'C'])

Parameter('K_egfr_egf', 1.2)
Rule('egfr_egf',
     [egfr(L=None), egf(R=None)],
     [egfr(L=1),    egf(R=1)],
     K_egfr_egf)
