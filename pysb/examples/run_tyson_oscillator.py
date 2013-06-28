# This creates model.odes which contains the math
from pysb.bng import generate_equations
from pysb.integrate import odesolve
from pysb.examples.tyson_oscillator import model
from numpy import *
import matplotlib.pyplot as plt

generate_equations(model)

t = linspace(0, 100, 10001)
x = odesolve(model, t)

#plot(t, x['CT'])  # Good validation of mass balance for cdc2, should be constant at 1
plot(t, x['YT'])
plot(t, x['M'])
show()
