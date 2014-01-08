# 1991, Tyson, Modeling the cell division cycle: cdc2 and cyclin interactions

from pysb import *
from pysb.macros import *

model = Model('oscillator')

Parameter('k1',  0.015)
Parameter('k2',  0)
Parameter('k3',  200)
Parameter('k4',  180)
Parameter('kp4', 0.018)
Parameter('k5',  0)
Parameter('k6',  1.0)
Parameter('k7',  0.6)
Parameter('k8',  1e12)
Parameter('k9',  1e6)

Monomer('cyclin', ['Y', 'b'], {'Y': ['U','P']})
Monomer('cdc2',   ['Y', 'b'], {'Y': ['U','P']})

# Rule 1
synthesize(cyclin(Y='U', b=None), k1)

# Rule 2
degrade(   cyclin(Y='U', b=None), k2)

# Rule 3
Rule('BindingAndPhosphoylation', cyclin(Y='U', b=None) + cdc2(Y='P', b=None) >> cyclin(Y='P', b=1) % cdc2(Y='P', b=1), k3)

# Rule 4
Rule('Activation', cyclin(Y='P', b=1) % cdc2(Y='P', b=1) >> cyclin(Y='P', b=1) % cdc2(Y='U', b=1), kp4)

# Rule 4'
Rule('Autocatalytic', cyclin(Y='P', b=1) % cdc2(Y='P', b=1) + cyclin(Y='P', b=2) % cdc2(Y='U', b=2) + cyclin(Y='P', b=2) % cdc2(Y='U', b=2) >>
                      cyclin(Y='P', b=1) % cdc2(Y='U', b=1) + cyclin(Y='P', b=2) % cdc2(Y='U', b=2) + cyclin(Y='P', b=2) % cdc2(Y='U', b=2), k4)

# Rule 5
Rule('Opposed', cyclin(Y='P', b=1) % cdc2(Y='U', b=1)  >> cyclin(Y='P', b=1) % cdc2(Y='P', b=1), k5)

# Rule 6
Rule('Dissociation', cyclin(Y='P', b=1) % cdc2(Y='U', b=1) >> cyclin(Y='P', b=None) + cdc2(Y='U', b=None), k6)

# Rule 7
degrade(cyclin(Y='P', b=None), k7)

# Rule 8 and 9
equilibrate(cdc2(Y='U', b=None), cdc2(Y='P', b=None), [k8, k9])

Observable("YT", cyclin())                                # Total Cyclin
Observable("CT", cdc2())                                  # Total CDC2
Observable("M",  cyclin(Y='P', b=1) % cdc2(Y='U', b=1) )  # Active Complex

# [C2] in Tyson
Parameter("cdc0", 1)
Initial(cdc2(Y='P', b=None), cdc0)

# [Y] in Tyson
Parameter('cyc0', 0.333333)
Initial(cyclin(Y='U', b=None), cyc0)



