""" Run the Extrinsic Apoptosis Reaction Model (EARM) using BioNetGen's
Hybrid-Particle Population (HPP) algorithm.

NFsim provides stochastic simulation without reaction network generation, 
allowing simulation of models with large (or infinite) reaction networks by 
keeping track of species counts. However, it can fail when the number of 
instances of a species gets too large (typically >200000). HPP circumvents 
this problem by allowing the user to define species with large instance 
counts as populations rather than NFsim particles.

This example runs the EARM 1.0 model with HPP, which fails to run on NFsim 
with the default settings due to large initial concentration coutns of 
several species. By assigning population maps to these species, we can run 
the simulation.

Reference: Hogg et al. Plos Comb Biol 2014
           https://doi.org/10.1371/journal.pcbi.1003544
"""
from pysb.examples.earm_1_0 import model
from pysb.simulator import BngSimulator
from pysb.simulator.bng import PopulationMap
from pysb import Parameter
import matplotlib.pyplot as plt
import numpy as np


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
    plt.ylabel('Population of %s' % name)

PARP, CPARP, Mito, mCytoC = [model.monomers[x] for x in
                             ['PARP', 'CPARP', 'Mito', 'mCytoC']]
klump = Parameter('klump', 10000, _export=False)
model.add_component(klump)

population_maps = [
    PopulationMap(PARP(b=None), klump),
    PopulationMap(CPARP(b=None), klump),
    PopulationMap(Mito(b=None), klump),
    PopulationMap(mCytoC(b=None), klump)
]

sim = BngSimulator(model, tspan=np.linspace(0, 20000, 101))
simres = sim.run(n_runs=20, method='nf', population_maps=population_maps)

trajectories = simres.all
tout = simres.tout

plot_mean_min_max('Bid_unbound')
plot_mean_min_max('PARP_unbound')
plot_mean_min_max('mSmac_unbound')
plot_mean_min_max('tBid_total')
plot_mean_min_max('CPARP_total')
plot_mean_min_max('cSmac_total')

plt.show()
