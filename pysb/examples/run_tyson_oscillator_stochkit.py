from pysb.examples.tyson_oscillator import model 
from pysb.tools.stochkit import run_stochkit
import numpy as np
import matplotlib.pyplot as plt

def plot_mean_min_max(name, title=None):
    x = np.array([tr[:][name] for tr in trajectories]).T
    if not title:
        title = name
    plt.figure(title)
    plt.plot(tout.T, x, '0.5', lw=2, alpha=0.25) # individual trajectories
    plt.plot(tout[0], x.mean(1), 'k--', lw=3, label="Mean")
    plt.plot(tout[0], x.min(1), 'b--', lw=3, label="Minimum")
    plt.plot(tout[0], x.max(1), 'r--', lw=3, label="Maximum")
    plt.legend(loc=0)
    plt.xlabel('Time')
    plt.ylabel('Population')

tspan = np.linspace(0, 50, 5001)
tout, trajectories = run_stochkit(model, tspan, n_runs=5, seed=None, algorithm="ssa", verbose=True)

plot_mean_min_max('__s0', str(model.species[0]))
plot_mean_min_max('YT')
plot_mean_min_max('M')

plt.show()
