
# This creates model.odes which contains the math
from pysb.bng import generate_equations
from pysb.integrate import odesolve
from pysb.examples.tropical_oscillator import model
from pylab import *

ion()

generate_equations(model)

t = linspace(0, 100, 10001)
x = odesolve(model, t, y0=[1.0, 0.33, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

#plot(t, x['CT'])  # Good validation of mass balance for cdc2

plot(t, x['YT']/x['CT'])
plot(t, x['M']/x['CT'])
show()


from sympy import symbols
k1 = symbols('k1', real=True)
k2 = symbols('k2', real=True)
k3 = symbols('k3', real=True)
k4 = symbols('k4', real=True)
kp4 = symbols('kp4', real=True)
k5 = symbols('k5', real=True)
k6 = symbols('k6', real=True)
k7 = symbols('k7', real=True)

s0 = symbols('s0', real=True)
s1 = symbols('s1', real=True)
s2 = symbols('s2', real=True)
s3 = symbols('s3', real=True)
s4 = symbols('s4', real=True)
s5 = symbols('s5', real=True)
s6 = symbols('s6', real=True)
s7 = symbols('s7', real=True)

# Patches required to work!!!
model.odes[1] = k1 - k2*s1 - k3*s1*s4                  # synthesize macro appears to have a bug
