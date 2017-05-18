from pysb import *

Model()
Monomer('X')
Monomer('A')
Monomer('B')

Parameter('X_0', 100)
Parameter('A_0', 1e5)
Parameter('B_0', 2e5)

Initial(X(), X_0)
Initial(A(), A_0)
Initial(B(), B_0)

Parameter('k1', 3e-7)
# Parameter('k2', 1e-4)

Parameter('k3', 1e-3)
Parameter('k4', 3.5)

Parameter('k2_n', 1e-4 / 3.)

Parameter('atol_k1', A_0.value * k1.value)
Parameter('btol_k3', B_0.value * k3.value)

# Rule('rule1', A() + X() + X() <> A() + X() + X() + X(), k1, k2_n)
# Rule('rule2', B() <> X() + B(), k3, k4)

Monomer('I')
Initial(I(), Parameter('I_0', 1))
Rule('rule3', X() + X() <> X() + X() + X(), atol_k1, k2_n)
Rule('rule4', I() <> X() + I(), btol_k3, k4)


Observable('X_total', X())
# Observable('A_total', A())
# Observable('B_total', B())
