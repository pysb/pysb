"""A trivial example of synthesis and degradation rules"""

from __future__ import print_function
from pysb import *

Model()

Monomer('A')

Parameter('kA_syn', 1e0)
Parameter('kA_deg', 1e-1)
Rule('synthesize_A', None >> A(), kA_syn)
Rule('degrade_A', A() >> None, kA_deg)

Parameter('A_0', 1.0)
Initial(A(), A_0)

Observable('A_total', A())

if __name__ == '__main__':
    print(__doc__, "\n", model)
    print("""
NOTE: This model code is designed to be imported and programatically
manipulated, not executed directly. The above output is merely a
diagnostic aid.""")
