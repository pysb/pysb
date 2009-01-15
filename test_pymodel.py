from Pysb import *


egf = Monomer('egf', ['L'])
egfr = Monomer('egfr', ['R', 'D', 'C'])

mp1 = egf.m(L=None)
mp2 = egfr.m(R=None)
mp3 = egfr.m(R=egf, D=egfr, C=None)

print egf
print egfr
print
print mp1
print mp2
print mp3
