"""Example of using an initial condition with a fixed amount.

With the amount of free F fixed, most of the free A will be able to bind and
form the A-F complex before equilibrium is reached. If free F were not fixed,
its lower initial amount would severely restrict how much complex could form.
See run_fixed_initial.py for a demonstration of this behavior.

"""

from __future__ import print_function
from pysb import *

Model()

Monomer('A', ['b'])
Monomer('F', ['b'])

Parameter('kf', 1.0)
Parameter('kr', 1.0)
Parameter('A_0', 100)
Parameter('F_0', 20)

Rule('A_bind_F', A(b=None) + F(b=None) | A(b=1) % F(b=1), kf, kr)

Initial(A(b=None), A_0)
Initial(F(b=None), F_0, fixed=True)

Observable('A_free', A(b=None))
Observable('F_free', F(b=None))
Observable('AF_complex', A(b=1) % F(b=1))


if __name__ == '__main__':
    print(__doc__, "\n", model)
    print("""
NOTE: This model code is designed to be imported and programatically
manipulated, not executed directly. The above output is merely a
diagnostic aid.""")
