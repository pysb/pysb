"""Demonstrate the effect of fixed initial conditions."""

import copy
import numpy as np
import matplotlib.pyplot as plt
from pysb.simulator import ScipyOdeSimulator

from pysb.examples.fixed_initial import model

n_obs = len(model.observables)
tspan = np.linspace(0, 0.5)

plt.figure(figsize=(8,4))

# Simulate the model as written, with free F fixed.
sim = ScipyOdeSimulator(model, tspan)
res = sim.run()
obs = res.observables.view(float).reshape(-1, n_obs)
plt.subplot(121)
plt.plot(res.tout[0], obs)
plt.ylabel('Amount')
plt.xlabel('Time')
plt.legend([x.name for x in model.observables], frameon=False)
plt.title('Free F fixed')

# Make a copy of the model and unfix the initial condition for free F.
model2 = copy.deepcopy(model)
model2.reset_equations()
model2.initials[1].fixed = False
sim2 = ScipyOdeSimulator(model2, tspan)
res2 = sim2.run()
obs2 = res2.observables.view(float).reshape(-1, n_obs)
plt.subplot(122)
plt.plot(res2.tout[0], obs2)
plt.xlabel('Time')
plt.legend([x.name for x in model.observables], frameon=False)
plt.title('Free F can be consumed')

plt.show()
