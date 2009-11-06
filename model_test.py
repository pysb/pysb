from pysb import *
import pysb.generator.bng as bng

Model('test')

Compartment('extracellular', dimension=3)
Compartment('cytoplasm', dimension=3)
Compartment('membrane', dimension=2, neighbors=[extracellular, cytoplasm])

Monomer('egf', 'R')
Monomer('egfr', ['L', 'D', 'C'])

Parameter('Kf_egfr_egf', 1.2)
Parameter('Kr_egfr_egf', 1.1)
Rule('egfr_egf',
     egfr(L=None) + egf(R=None) <>
     egfr(L=1)    * egf(R=1),
     Kf_egfr_egf, Kr_egfr_egf)

test.observe('free_egf', egf(R=None))
test.observe('free_egfr', egfr(L=None))
test.observe('bound', egf(R=1) * egfr(L=1))

gen = bng.BngGenerator(model=test)
bng_content = gen.content
bng_content += """
begin species
  egf(R)       6.0
  egfr(L,D,C)  10.0
end species
"""

if __name__ == '__main__':
    print bng_content
    print "begin actions"
    print "  generate_network({overwrite=>1});"
    print "end actions"

