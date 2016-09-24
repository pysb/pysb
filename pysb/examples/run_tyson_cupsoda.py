import matplotlib.pyplot as plt
import numpy as np

from pysb.examples.tyson_oscillator import model
from pysb.simulator import CupSodaSolver

tspan = np.linspace(0, 500, 501)

solver = CupSodaSolver(model, tspan, atol=1e-12, rtol=1e-6, max_steps=20000,
                       verbose=False)

n_sims = 100

# Rate constants
param_values = np.ones((n_sims, len(model.parameters)))
for i in range(len(param_values)):
    for j in range(len(param_values[i])):
        param_values[i][j] *= model.parameters[j].value

# Initial concentrations
y0 = np.zeros((n_sims, len(model.species)))
for i in range(len(y0)):
    for ic in model.initial_conditions:
        for j in range(len(y0[i])):
            if str(ic[0]) == str(model.species[j]):
                y0[i][j] = ic[1].value
                break

yfull = solver.run(param_values=param_values, y0=y0)

# Plot the results of the first simulation
plt.plot(tspan, np.array(yfull.observables)['YT'].T, lw=2, label='YT',
         color='b')
plt.plot(tspan, np.array(yfull.observables)['M'].T, lw=2, label='M', color='g')
plt.legend(loc=0)
plt.ylim(ymin=1)
plt.ylabel('molecules')
plt.xlabel('time')
plt.show()
