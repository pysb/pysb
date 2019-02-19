from pysb.examples.michment import model
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Construct initials dict
mult = np.linspace(0.8, 1.2, 11)
matrix = [mult*ic.value.value for ic in model.initials]
initials = {}
for i, ic in enumerate(model.initials):
    initials[ic.pattern] = []
    for tup in itertools.product(*matrix): # Cartesian product
        initials[ic.pattern].append(tup[i])

tspan = np.linspace(0, 50, 501)
sim = ScipyOdeSimulator(model, tspan, verbose=False)
trajectories = sim.run(initials=initials).all

x = np.array([tr['Product'] for tr in trajectories]).T

plt.plot(tspan, x.mean(axis=1), 'b', lw=3, label="Product")
plt.plot(tspan, x.max(axis=1), 'b--', lw=2, label="min/max")
plt.plot(tspan, x.min(axis=1), 'b--', lw=2)
plt.xlabel('time')
plt.ylabel('concentration')
plt.legend(loc='upper left')

plt.show()
