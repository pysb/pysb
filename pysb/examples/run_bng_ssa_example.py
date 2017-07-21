import matplotlib.pyplot as plt
import numpy as np
from pysb.simulator.bng import BngSimulator
from kinase_cascade import model

# We will integrate from t=0 to t=40
t = np.linspace(0, 40, 50)

# Simulate the model
print("Simulating...")
sim = BngSimulator(model)
x = sim.run(tspan=t, verbose=False, n_runs=5, method='ssa')
tout = x.tout
y = np.array(x.observables)
plt.plot(tout.T, y['ppMEK'].T)
plt.show()
