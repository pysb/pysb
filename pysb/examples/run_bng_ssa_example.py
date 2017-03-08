import matplotlib.pyplot as plt
import numpy as np
from pysb.simulator.bng_ssa import BioNetGenSSASimulator
from kinase_cascade import model

# We will integrate from t=0 to t=40
t = np.linspace(0, 40, 50)
# Simulate the model
print("Simulating...")
sim = BioNetGenSSASimulator(model)
x = sim.run(tspan=t, verbose=False, n_sim=10)
tout = x.tout
y = np.array(x.observables)
plt.plot(tout.T, y['ppMEK'].T)
plt.show()
