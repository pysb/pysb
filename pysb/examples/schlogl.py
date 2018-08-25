from pysb import *

Model()
Monomer('X')

Parameter('X_0', 100)
Parameter('A_0', 1e5)
Parameter('B_0', 2e5)

Parameter('k1', 3e-7)
Parameter('k3', 1e-3)

Parameter('k4', 3.5)

Parameter('k2_n', 1e-4 / 3.)

Parameter('atol_k1', A_0.value * k1.value)
Parameter('btol_k3', B_0.value * k3.value)

Monomer('I')
Initial(X(), X_0)
Initial(I(), Parameter('I_0', 1))
Rule('rule3', X() + X() | X() + X() + X(), atol_k1, k2_n)
Rule('rule4', I() | X() + I(), btol_k3, k4)

Observable('X_total', X())
