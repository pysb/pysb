# -*- coding: utf-8 -*-
""" Schlögl model

Schlögl, F. Z. Physik (1972) 253: 147. https://doi.org/10.1007/BF01379769
"""
from pysb import *

Model()

Parameter('X_0', 100)
Parameter('k4', 3.5)

Parameter('k2_n', 1e-4 / 3.)

Parameter('atol_k1', 1e5 * 3e-7)
Parameter('btol_k3', 2e5 * 1e-3)

Monomer('X')
Monomer('I')
Initial(X(), X_0)
Initial(I(), Parameter('I_0', 1))
Rule('rule3', X() + X() | X() + X() + X(), atol_k1, k2_n)
Rule('rule4', I() | X() + I(), btol_k3, k4)

Observable('X_total', X())
