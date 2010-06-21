from pysb import *
import logging

yeahlogging.basicConfig()
complog = logging.getLogger("compartments_file")
complog.setLevel(logging.DEBUG)

complog.debug("starting Model")
Model('test')

complog.debug("setting Compartment")
# Compartment('eCell',     dimension=3, size=extraSize, parent=None)
# Compartment('membrane',  dimension=2, size=memSize,   parent=extra)
# Compartment('cytoplasm', dimension=3, size=cytoSize,  parent=membrane)



Monomer('egf', 'R', {'R':['up', 'down']})
Monomer('egfr', ['L', 'D', 'C'], {'C':['on', 'off']})

Parameter('K_egfr_egf', 1.2)
Rule('egfr_egf',
     egfr(L=None) + egf(R=None) >>
     egfr(L=1)    * egf(R=1), K_egfr_egf)

print egf
print egfr

print "\ntesting error checking..."

fail = False
try:
    Monomer('M', ['a', 'b', 'c', 'a', 'c'])
    fail = True
except:
    pass
if fail:
    raise Exception("Didn't throw expected exception")

fail = False
try:
    egfr(L=None, x=None)
    fail = True
except:
    pass
if fail:
    raise Exception("Didn't throw expected exception")

fail = False
try:
    egfr('x')
    fail = True
except:
    pass
if fail:
    raise Exception("Didn't throw expected exception")

fail = False
try:
    egfr(D=[egfr, 'x'])
    fail = True
except:
    pass
if fail:
    raise Exception("Didn't throw expected exception")

print "\tsuccess!"
