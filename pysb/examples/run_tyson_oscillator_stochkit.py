from pysb.examples.tyson_oscillator import model 
from pysb.tools.pysb2gillespy import translate
from gillespy import StochKitSolver
import numpy as np
import matplotlib.pyplot as plt

trajectories = StochKitSolver.run(translate(model,verbose=True), t=50, number_of_trajectories=10, increment=0.01)

t = np.array(trajectories[0][:,0])
y = np.array([x[:,1] for x in trajectories]).T

plt.figure(str(model.species[1]))
# plot individual trajectories
plt.plot(t, y, '0.5', lw=2, alpha=0.25)
# plot mean
plt.plot(t, y.mean(1), 'k--', lw=3, label="Mean")
# plot min & max
plt.plot(t, y.min(1), 'b--', lw=3, label="Minimum")
plt.plot(t, y.max(1), 'r--', lw=3, label="Maximum")

plt.legend(loc=0)
plt.xlabel('Time')
plt.ylabel('Population')

plt.show()
