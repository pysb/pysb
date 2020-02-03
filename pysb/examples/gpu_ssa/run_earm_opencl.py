"""
Example of using OpenCLSSASimulator for SSA simulations.


This example uses the earm_1_0 model.
We recommend using a dedicated GPU without a display attached at there can
be time out. For testing, one can change the tspan = np.linspace(0,5000,101).

"""

import matplotlib.pyplot as plt
import numpy as np
from pysb.examples.earm_1_0 import model
from pysb.simulator import OpenCLSSASimulator, ScipyOdeSimulator

if __name__ == "__main__":
    n_sim = 100
    tspan = np.linspace(0, 5000, 101)
    name = 'tBid_total'
    sim = OpenCLSSASimulator(model, tspan=tspan, verbose=True)
    traj = sim.run(tspan=tspan, number_sim=n_sim)

    result = np.array(traj.observables)[name]

    x = np.array([tr[:] for tr in result]).T

    plt.plot(tspan, x, '0.5', lw=2, alpha=0.25)  # individual trajectories
    plt.plot(tspan, x.mean(axis=1), 'b', lw=3, label='mean')
    plt.plot(tspan, x.max(axis=1), 'k--', lw=2, label="min/max")
    plt.plot(tspan, x.min(axis=1), 'k--', lw=2)

    sol = ScipyOdeSimulator(model, tspan)
    traj = sol.run()

    plt.plot(tspan, np.array(traj.observables)[name], label='ode', color='red')

    plt.legend(loc=0)
    plt.tight_layout()
    out_name = 'earm_ssa_example.png'
    plt.savefig(out_name)
    plt.close()
