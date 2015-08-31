"""Demonstration of the move_connected keyword for Rules
"""

from __future__ import print_function
from pysb import *

Model()

Monomer('A', ['b'])
Monomer('B', ['a'])

# One main 3d compartment and two 2d membranes inside it
Parameter('Vmain', 1)
Parameter('Vx', 1)
Parameter('Vy', 1)
Compartment('Main', None, 3, Vmain)
Compartment('X', Main, 2, Vx)
Compartment('Y', Main, 2, Vy)

# A and B, both embedded in membrane X, bind reversibly
Parameter('Kab_f', 1)
Parameter('Kab_r', 1)
Rule('Ax_bind_Bx', A(b=None) ** X + B(a=None) ** X <> A(b=1) ** X % B(a=1) ** X,
     Kab_f, Kab_r)

# The A:B complex is transported back and forth from X to Y
Parameter('Ktrans_f', 1)
Parameter('Ktrans_r', 1)
# move_connected is required or B will be "left behind" and BNG will complain
# (change move_connected to False and run pysb.tools.export_bng_net on this file
# and watch for the WARNING line in the output log)
Rule('ABx_trans_y', A(b=ANY) ** X <> A(b=ANY) ** Y,
     Ktrans_f, Ktrans_r, move_connected=True)

Parameter('ABx_0', 1)
Initial(A(b=1)**X % B(a=1)**X, ABx_0)

if __name__ == '__main__':
    print(__doc__, "\n", model)
    print("""
NOTE: This model code is designed to be imported and programatically
manipulated, not executed directly. The above output is merely a
diagnostic aid.""")
