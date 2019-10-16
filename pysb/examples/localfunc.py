# Based on the BioNetGen model 'localfunc.bngl'
# https://github.com/RuleWorld/bionetgen/blob/master/bng2/Models2/localfunc.bngl
#
# This model demonstrates the use of MultiState and local functions in PySB
#
# Requires Python 3.x or greater (will give a SyntaxError on Python 2.7)

from pysb import Model, Monomer, Parameter, Expression, Rule, \
    Observable, Initial, Tag, MultiState

Model()

Monomer('A', ['b', 'b', 'b'])
Monomer('B', ['a'])
Monomer('C')

Parameter('kp', 0.5)
Parameter('km', 0.1)
Parameter('k_synthC', 1e3)
Parameter('k_degrC', 0.5)
Parameter('Ab_b_b_0', 1.0)
Parameter('Ba_0', 3.0)
Parameter('C_0', 0.0)

Observable('Atot', A())
Observable('Btot', B())
Observable('Ctot', C())
Observable('AB0', A(b=MultiState(None, None, None)), match='species')
Observable('AB1', A(b=MultiState(1, None, None)) % B(a=1), match='species')
Observable('AB2', A(b=MultiState(1, 2, None)) % B(a=1) % B(a=2),
           match='species')
Observable('AB3', A(b=MultiState(1, 2, 3)) % B(a=1) % B(a=2) % B(a=3),
           match='species')
Observable('AB_motif', A(b=1) % B(a=1))

Tag('x')

Expression('f_synth', k_synthC * AB_motif(x) ** 2)

# A synthesizes C with rate dependent on bound B
Rule('_R1', A() @ x >> A() @ x + C(), f_synth)

# A binds B
Rule('_R2', A(b=None) + B(a=None) | A(b=1) % B(a=1), kp, km)

# degradation of C
Rule('_R3', C() >> None, k_degrC)

Initial(A(b=MultiState(None, None, None)), Ab_b_b_0)
Initial(B(a=None), Ba_0)
Initial(C(), C_0)
