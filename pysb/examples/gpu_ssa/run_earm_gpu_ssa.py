import logging

import matplotlib.pyplot as plt
import numpy as np

from pysb.examples.earm_1_0 import model
from pysb.logging import setup_logger
from pysb.simulator.opencl_ssa import OpenCLSimulator
from pysb.simulator.scipyode import ScipyOdeSimulator

setup_logger(logging.INFO)

obs = ['cSmac_total', 'tBid_total', 'CPARP_total',
       'Bid_unbound', 'PARP_unbound', 'mSmac_unbound']


def run(n_sim=1000):
    tspan = np.linspace(0, 20000, 101)
    name = 'tBid_total'

    sim = OpenCLSimulator(model, tspan=tspan, verbose=True, device='cpu')
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
    plt.savefig('example_ssa_earm.png', dpi=200, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    n_sim = 10
    run(n_sim)
