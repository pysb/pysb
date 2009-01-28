from Pysb import *

Compartment('membrane', dimension=2)

Monomer('egf', 'R')
Monomer('egfr', ['L', 'D', 'C'])

Parameter('K_egfr_egf', 1.2)
Rule('egfr_egf',
     [egfr(L=None), egf(R=None)],
     [egfr(L=1),    egf(R=1)],
     K_egfr_egf)

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
