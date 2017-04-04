import matplotlib.pyplot as plt
import numpy as np
from pysb.simulator.bng_ssa import BngSimulator
from kinase_cascade import model
# from bax_pore import  model

# We will integrate from t=0 to t=40
t = np.linspace(0, 40, 50)
# Simulate the model
print("Simulating...")
sim = BngSimulator(model, cleanup=False)
x = sim.run(tspan=t, verbose=False, n_sim=5, cleanup=False, method='ssa')
tout = x.tout
y = np.array(x.observables)
# plt.plot(tout.T, y['BAX4'].T)
plt.plot(tout.T, y['ppMEK'].T)
plt.show()
