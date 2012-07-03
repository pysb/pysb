from pysb import *
import pysb.integrate as pint
from pylab import *

Model()

Monomer('A')
Monomer('B')

Parameter('kA_syn', 1e0)
Parameter('kA_deg', 1e-1)
Parameter('kB_syn', 2e0)
Parameter('kB_deg', 2e-1)

Rule('synthesize_A', None >> A(), kA_syn)
Rule('degrade_A', A() >> None, kA_deg)
Rule('synthesize_B', None >> B(), kB_syn)
Rule('degrade_B', B() >> None, kB_deg)

Parameter('A_0', 1.0)
Parameter('B_0', 1.0)
Initial(A(), A_0)
Initial(B(), B_0)

Observable('A_total', A())
Observable('B_total', B())


def main():
    t = linspace(0, 60)
    y = pint.odesolve(model, t)
    print y.A_total[-1], y.B_total[-1]
    plot(t, y.A_total)
    plot(t, y.B_total)
    legend([o.name for o in model.observables])
    show()

if __name__ == '__main__':
    main()
