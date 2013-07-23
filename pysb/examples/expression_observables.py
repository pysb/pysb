"""Simple model demonstrating expressions that reference observables."""

from pysb import *

Model()

Monomer('Bax', ['conf'], {'conf': ['c0', 'c1', 'c2']})

Initial(Bax(conf='c0'), Parameter('Bax_0', 1))

Parameter('k1', 1)
Parameter('k2', 1)
Parameter('c0_scaling', 0)
Parameter('c1_scaling', 2)
Parameter('c2_scaling', 5)

Rule('c0_to_c1', Bax(conf='c0') >> Bax(conf='c1'), k1)
Rule('c1_to_c2', Bax(conf='c1') >> Bax(conf='c2'), k2)

Observable('Bax_c0', Bax(conf='c0'))
Observable('Bax_c1', Bax(conf='c1'))
Observable('Bax_c2', Bax(conf='c2'))

Expression('NBD_signal',
           Bax_c0 * c0_scaling + Bax_c1 * c1_scaling + Bax_c2 * c2_scaling)
