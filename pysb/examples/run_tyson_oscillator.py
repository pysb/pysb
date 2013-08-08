# This creates model.odes which contains the math
from pysb.bng import generate_equations
from pysb.integrate import odesolve
from pysb.examples.tyson_oscillator import model
from numpy import *
from pysb.tools.tropicalize import *
import matplotlib.pyplot as plt
from pysb.tools.varsens import varsens

generate_equations(model)

t = linspace(0, 100, 10001)
x = odesolve(model, t)
ref = x

#plot(t, x['CT'])  # Good validation of mass balance for cdc2, should be constant at 1
plt.plot(t, x['YT'])
plt.plot(t, x['M'])
plt.show()

m = Tropical(model)
m.tropicalize(t, verbose=True)

names = ['k1', 'k3', 'k4', 'kp4', 'k6', 'k7', 'k8', 'k9']
values = array([model.parameters[nm].value for nm in names])
scaling = [values-0.2*values, values+0.2*values] # 20% around values

def osc_objective(params):
    for i in range(len(params)):
        model.parameters[names[i]].value = params[i]
    x = odesolve(model, t)
    return sum((x['YT'] - ref['YT'])**2)

v = varsens(osc_objective, len(params), 1024, scaling)