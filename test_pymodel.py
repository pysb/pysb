from Pysb import *


egf = Monomer('egf', 'R')
egfr = Monomer('egfr', ['L', 'D', 'C'])

egfr.m(L=egf, D=[egfr,egfr], C=None)
egfr.m(L=1)

K_egfr_egf = Parameter('K_egfr_egf', 1.2)
r_egfr_egf = Rule('egfr_egf',
                  [egfr.m(L=None), egf.m(R=None)],
                  [egfr.m(L=1),    egf.m(R=1)],
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
    egfr.m(L=None, x=None)
    fail = True
except:
    pass
if fail:
    raise Exception("Didn't throw expected exception")

fail = False
try:
    egfr.m('x')
    fail = True
except:
    pass
if fail:
    raise Exception("Didn't throw expected exception")

fail = False
try:
    egfr.m(D=[egfr, 'x'])
    fail = True
except:
    pass
if fail:
    raise Exception("Didn't throw expected exception")

print "\tsuccess!"
