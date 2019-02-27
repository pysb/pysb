from pysb.examples.michment import model
from pysb.simulator.cupsoda import run_cupsoda
import numpy as np
import matplotlib.pyplot as plt
import itertools


def run():
    # factors to multiply the values of the initial conditions
    multipliers = np.linspace(0.8, 1.2, 11)
    # 2D array of initial concentrations
    initial_concentrations = [
        multipliers * ic.value.value for ic in model.initials
    ]
    # Cartesian product of initial concentrations
    cartesian_product = itertools.product(*initial_concentrations)
    # the Cartesian product object must be cast to a list, then to a numpy array
    # and transposed to give a (n_species x n_vals) matrix of initial concentrations
    initials_matrix = np.array(list(cartesian_product)).T
    # we can now construct the initials dictionary
    initials = {
        ic.pattern: initials_matrix[i] for i, ic in enumerate(model.initials)
    }
    # simulation time span and output points
    tspan = np.linspace(0, 50, 501)
    # run_cupsoda returns a 3D array of species and observables trajectories
    trajectories = run_cupsoda(model, tspan, initials=initials,
                               integrator_options={'atol': 1e-10, 'rtol': 1e-4},
                               verbose=True)
    # extract the trajectories for the 'Product' into a numpy array and
    # transpose to aid in plotting
    x = np.array([tr['Product'] for tr in trajectories]).T
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
