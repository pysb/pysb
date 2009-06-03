from Pysb import *
import generator.bng as bng

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

gen = bng.BngGenerator(model=test)
print gen.content
