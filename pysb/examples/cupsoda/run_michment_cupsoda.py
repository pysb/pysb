from pysb.examples.michment import model
from pysb.simulator.cupsoda import CupSodaSimulator
import numpy as np
import matplotlib.pyplot as plt

tspan = np.linspace(0, 50, 501)
sim = CupSodaSimulator(model, tspan, atol=1e-10, rtol=1e-4, 
                       verbose=False)

et_ref = model.parameters['Etot'].value
s0_ref = model.parameters['S0'].value

initials = []
for Etot in np.linspace(0.8, 1.2, 21):
    for S0 in np.linspace(0.8, 1.2, 21):
        initials.append( [Etot*et_ref, S0*s0_ref] + 
                         [0]*(len(model.species)-2) )

trajectories = sim.run(initials=initials).observables

x = np.array([tr[:]['Product'] for tr in trajectories]).T
 
plt.plot(tspan, x.mean(axis=1), 'b', lw=3, label="Product")
plt.plot(tspan, x.max(axis=1), 'b--', lw=2, label="min/max")
plt.plot(tspan, x.min(axis=1), 'b--', lw=2)
plt.xlabel('time')
plt.ylabel('concentration')
plt.legend(loc='upper left')
 
plt.show()
