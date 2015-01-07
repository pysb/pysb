"""A simple three-species chemical kinetics system known as "Robertson's
example", as presented in:

H. H. Robertson, The solution of a set of reaction rate equations, in Numerical
Analysis: An Introduction, J. Walsh, ed., Academic Press, 1966, pp. 178-182.
"""

# This is a simple system often used to study stiffness in systems of
# differential equations.  It doesn't leverage the power of rules-based modeling
# or pysb, but it's a useful small model for purposes of experimentation.
#
# A brief report addressing issues of stiffness encountered in numerical
# integration of Robertson's example can be found here:
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.53.8603&rep=rep1&type=pdf
#
# The chemical model is as follows:
#
#      Reaction        Rate
#   ------------------------
#       A -> B         0.04
#      2B -> B + C     3.0e7
#   B + C -> A + C     1.0e4
#
# The resultant system of differential equations is:
#
# y1' = -0.04 * y1 + 1.0e4 * y2 * y3
# y2' =  0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2^2
# y3' =                                3.0e7 * y2^2
#
# If you run this script directly, it will generate the equations and print them
# in a form very closely resembling that given above.  See also run_robertson.py
# which integrates the system and plots the trajectories.


from __future__ import print_function
from pysb import *

Model()

Monomer('A')
Monomer('B')
Monomer('C')

#       A -> B         0.04
Rule('A_to_B', A() >> B(), Parameter('k1', 0.04))
#      2B -> B + C     3.0e7
Rule('BB_to_BC', B() + B() >> B() + C(), Parameter('k2', 3.0e7))
#   B + C -> A + C     1.0e4
Rule('BC_to_AC', B() + C() >> A() + C(), Parameter('k3', 1.0e4))

# The system is known to be stiff for initial values A=1, B=0, C=0
Initial(A(), Parameter('A_0', 1.0))
Initial(B(), Parameter('B_0', 0.0))
Initial(C(), Parameter('C_0', 0.0))

# Observe total amount of each monomer
Observable('A_total', A())
Observable('B_total', B())
Observable('C_total', C())


if __name__ == '__main__':
    from pysb.bng import generate_equations

    # This creates model.odes which contains the math
    generate_equations(model)

    # Build a symbol substitution mapping to help the math look like the
    # equations in the comment above, instead of showing the internal pysb
    # symbol names
    # ==========
    substitutions = {}
    # Map parameter symbols to their values
    substitutions.update((p.name, p.value) for p in model.parameters)
    # Map species variables sI to yI+1, e.g. s0 -> y1
    substitutions.update(('s%d' % i, 'y%d' % (i+1))
                         for i in range(len(model.odes)))

    print(__doc__, "\n", model, "\n")

    # Iterate over each equation
    for i, eq in enumerate(model.odes):
        # Perform the substitution using the sympy 'subs' method and the
        # mappings we built above
        eq_sub = eq.subs(substitutions)
        # Display the equation
        print('y%d\' = %s' % (i+1, eq_sub))

    print("""
NOTE: This model code is designed to be imported and programatically
manipulated, not executed directly. The above output is merely a
diagnostic aid. Please see run_robertson.py for example usage.""")
