"""Produce contact map for Figure 5D from the PySB publication"""

from __future__ import print_function
from pysb import *
from earm import lopez_modules
from earm import albeck_modules
from pysb.kappa import contact_map
import os

# A variant version of Lopez embedded with no Smac/CytoC

Model()

lopez_modules.momp_monomers()
Observable('aBax', Bax(state='A'))
Observable('cSmac', Smac(state='C'))

# The specific MOMP model to use
lopez_modules.embedded(do_pore_transport=False)

# Set Bid initial condition to be tBid
model.update_initial_condition_pattern(Bid(state='U', bf=None),
                                       Bid(state='T', bf=None))
# Get rid of CytoC and Smac
model.parameters['Smac_0'].value = 0
model.parameters['CytoC_0'].value = 0

# Put some Noxa at the membrane
Initial(Noxa(state='M', bf=None), Parameter('mNoxa_0', 1))
Initial(Bad(state='M', bf=None), Parameter('mBad_0', 1))


# Generate the contact map
output_dir = os.path.abspath(os.path.dirname(__file__))
contact_map(model, output_dir, 'fig5d')
try:
    # Remove the extra output files we don't need
    os.unlink('fig5d.ka')
    os.unlink('fig5d_cm.jpg')
except OSError:
    pass
print()
print("Generated fig5d_cm.dot in", output_dir)
