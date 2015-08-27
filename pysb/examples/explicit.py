"""Example of how to write a model without using SelfExporter.
"""

from __future__ import print_function
from pysb import *
import pysb.core

# disable SelfExporter from doing its thing
pysb.core.SelfExporter.do_export = False

model = Model('explicit')

L = Monomer('L', ['r'])
R = Monomer('R', ['l'])
model.add_component(L)
model.add_component(R)

kf = Parameter('kf', 1e-6)
L_binds_R = Rule('L_binds_R', L(r=None) + R(l=None) >> L(r=1) % R(l=1), kf)
model.add_component(kf)
model.add_component(L_binds_R)

L_0 = Parameter('L_0', 1e2)
R_0 = Parameter('R_0', 1e4)
model.add_component(L_0)
model.add_component(R_0)
model.initial(L(r=None), L_0)
model.initial(R(l=None), R_0)

if __name__ == '__main__':
    print(__doc__, "\n", model)
    print("""
NOTE: This model code is designed to be imported and programatically
manipulated, not executed directly. The above output is merely a
diagnostic aid.""")
