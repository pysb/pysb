from pysb.examples.michment import model
from pysb.simulator.cutauleaping import CuTauLeapingSimulator
import numpy as np
import matplotlib.pyplot as plt


def run():

    # simulation time span and output points
    tspan = np.linspace(0, 50, 101)

    sim = CuTauLeapingSimulator(model, tspan)
    trajectories = sim.run(number_sim=100)
    trajectories = np.array(trajectories.observables)['Product']
    # extract the trajectories for the 'Product' into a numpy array and
    # transpose to aid in plotting
    x = np.array([np.array(tr) for tr in np.array(trajectories)]).T
    # plot the mean, minimum, and maximum concentrations at each time point
    plt.plot(tspan, x.mean(axis=1), 'b', lw=3, label="Product")
    plt.plot(tspan, x.max(axis=1), 'b--', lw=2, label="min/max")
    plt.plot(tspan, x.min(axis=1), 'b--', lw=2)
    # define the axis labels and legend
    plt.xlabel('time')
    plt.ylabel('concentration')
    plt.legend(loc='upper left')
    # show the plot
    plt.show()


if __name__ == '__main__':
    run()
