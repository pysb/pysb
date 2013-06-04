
# This creates model.odes which contains the math
from pysb.bng import generate_equations
from pysb.integrate import odesolve
from pysb.examples.tropical_oscillator import model
from pylab import *

ion()

generate_equations(model)

t = linspace(0, 100, 10001)
x = odesolve(model, t)

#plot(t, x['CT'])  # Good validation of mass balance for cdc2
plot(t, x['YT']/x['CT'])
plot(t, x['M']/x['CT'])
show()
