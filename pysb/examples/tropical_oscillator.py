# 1991, Tyson, Modeling the cell division cycle: cdc2 and cyclin interactions

from pysb import *
from pysb.macros import *

model = Model('oscillator')

Parameter('k1',  0.015)
#Parameter('k2',  0)
Parameter('k3',  200)
Parameter('k4',  10)
Parameter('kp4', 0.018)
#Parameter('k5',  0)
Parameter('k6',  1.0)
Parameter('k7',  0.6)
Parameter('k8',  1e6)
Parameter('k9',  1e3)

Monomer('cyclin', ['Y', 'b'], {'Y': ['U','P']})
Monomer('cdc2',   ['Y', 'b'], {'Y': ['U','P']})

synthesize(cyclin(Y='U', b=None), k1)
#degrade(cyclin(Y='U', b=None), k2)

# One-way binding and phosophylation in one step (?)
Rule('Step3', cyclin(Y='U', b=None) + cdc2(Y='P',b=None) >> cyclin(Y='P', b=1) % cdc2(Y='P', b=1), k4)

#equilibrate(cyclin(Y='P', b=1) % cdc2(Y='P', b=1), cyclin(Y='P', b=1) % cdc2(Y='U', b=1), [kp4, k5])
Rule('Activation', cyclin(Y='P', b=1) % cdc2(Y='P', b=1) >> cyclin(Y='P', b=1) % cdc2(Y='U', b=1), kp4)

Rule('Autocatalytic', cyclin(Y='P', b=1) % cdc2(Y='P', b=1) + cyclin(Y='P', b=2) % cdc2(Y='U', b=2) >>
                      cyclin(Y='P', b=1) % cdc2(Y='U', b=1) + cyclin(Y='P', b=2) % cdc2(Y='U', b=2), k4)

Rule('Dissociation', cyclin(Y='P', b=1) % cdc2(Y='U', b=1) >> cyclin(Y='P', b=None) + cdc2(Y='U', b=None), k6)

degrade(cyclin(Y='P', b=None), k7)

equilibrate(cdc2(Y='U', b=None), cdc2(Y='P', b=None), [k8, k9])

Observable("YT", cyclin())
Observable("CT", cdc2())
Observable("M",  cyclin(Y='P', b=1) % cdc2(Y='U', b=1) )

Parameter("cdc0", 1)
Initial(cdc2(Y='U', b=None), cdc0)

Parameter("cyc0", 1/4)
Initial(cyclin(Y='U', b=None), cyc0)


# This creates model.odes which contains the math
from pysb.bng import generate_equations
from pysb.integrate import odesolve
from pylab import *

generate_equations(model)

t = linspace(0, 100, 10001)
x = odesolve(oscillator, t)

plot(t, x['YT'])
show()

plot(t, x['M'])
show()

