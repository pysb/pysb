import numpy as np
from pysb.simulator.scipyode import ScipyOdeSimulator
from pysb.tools.sensitivity_analysis import \
    InitialsSensitivity
from pysb.examples.tyson_oscillator import model


tspan = np.linspace(0, 200, 5001)


def obj_func_cell_cycle(trajectory):
    """
    Calculate the frequency of the Y3

    Parameters
    ----------
    trajectory : vector_like
        Simulation trajectory for the Y3 observable

    Returns
    -------
    local_freq : float
        frequency value of Y3 observable
    """
    timestep = tspan[:-1]
    y = trajectory[:-1] - trajectory[1:]
    freq = 0
    local_times = []
    prev = y[0]
    # easy calculation of frequency,
    # find two positions where slope changes
    for n in range(1, len(y)):
        if y[n] > 0 > prev:
            local_times.append(timestep[n])
            freq += 1
        prev = y[n]

    local_times = np.array(local_times)
    local_freq = np.average(local_times)/len(local_times)*2
    return local_freq


def run():
    # The observable of the model
    observable = 'Y3'
    # The values of each initial concentration to samples
    # These values will be per initial concentration
    vals = [.8, 1.0, 1.2]

    # need to create a solver to run the model
    solver = ScipyOdeSimulator(model, tspan)

    # initialize the sensitivity class
    sens = InitialsSensitivity(
        values_to_sample=vals,
        observable=observable,
        objective_function=obj_func_cell_cycle,
        solver=solver
    )

    # runs the function, can pass save_name and out_dir to save sens matrices
    sens.run()

    # some sample plotting commands to help view the sensitivities
    sens.create_individual_pairwise_plots(save_name='pairwise_individual',
                                          out_dir='tyson_sensitivity')
    sens.create_plot_p_h_pprime(save_name='matrices',
                                out_dir='tyson_sensitivity')
    # creates a heatplot of all initial concentration in a mirrored grid
    # also decomposed heatplot into single initial concentration species
    sens.create_boxplot_and_heatplot(save_name='tyson_sensitivity',
                                     out_dir='tyson_sensitivity',
                                     show=False)
    print("Results saved in tyson_sensitivity directory")


if __name__ == '__main__':
    run()
