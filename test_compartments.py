from pysb import *
import logging

logging.basicConfig()
complog = logging.getLogger("compartments_file")
complog.setLevel(logging.DEBUG)

complog.debug("starting Model")
Model('comptest')

complog.debug("setting Compartment")
Compartment('eCell',     dimension=3, size=1,   parent=None)
Compartment('ECmembrane',  dimension=2, size=1.0, parent=eCell)
Compartment('cytoplasm', dimension=3, size=1.0, parent=ECmembrane)


complog.debug("setting Monomers")
Monomer('egf', 'R', {'R':['up', 'down']}, compartment = eCell)
Monomer('egfr', ['L', 'D', 'C'], {'C':['on', 'off']}, compartment = ECmembrane)
Monomer('shc', ['L', 'A'], {'A':['on', 'off']}, compartment = cytoplasm)

Parameter('K_egfr_egf_F', 1.2)
Parameter('K_egfr_egf_R', 1.2)
Rule('egfr_egf',
     egfr(L=None) + egf(R=None) <>
     egfr(L=1)    ** egf(R=1), K_egfr_egf_F, K_egfr_egf_R)

print egf
print egfr
print shc
print eCell
print ECmembrane
print cytoplasm
