from Pysb import *


egf = Monomer('egf', 'L')
egfr = Monomer('egfr', ['R', 'D', 'C'])

mp1 = egf.m(L=None)
mp2 = egfr.m(R=None)
mp3 = egfr.m(R=egf, D=[egfr,egfr], C=None)
egfr.m(R=1)

print egf
print egfr
print
print mp1
print mp2
print mp3

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
