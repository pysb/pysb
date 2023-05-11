"""
Example of using CudaSSASimulator for SSA simulations.


This example uses the schlogl model.
Users can run `vary_x` to create an image demonstrating the bimodal behavior of
the model.

"""
import matplotlib.pyplot as plt
import numpy as np
from pysb.examples.schloegl import model
from pysb.simulator import CudaSSASimulator, ScipyOdeSimulator


def run(n_sim=100, x_0=100):
    """

    Parameters
    ----------
    n_sim
    x_0

    Returns
    -------

    """
    obs_name = 'X_total'
    tspan = np.linspace(0, 100, 101)
    x_0 = int(x_0)
    model.parameters['X_0'].value = x_0
    savename = 'schloegl_{}'.format(int(x_0))

    # create simulator and run simulations
    traj = CudaSSASimulator(model).run(tspan, number_sim=n_sim)
    x = traj.dataframe[obs_name].unstack(0).values

    # create line traces
    plt.figure()
    plt.subplot(121)
    plt.plot(tspan, x, '0.5', lw=2, alpha=0.25)  # individual trajectories
    plt.plot(tspan, x.mean(1), 'k-*', lw=3, label="Mean")
    plt.plot(tspan, x.min(1), 'b--', lw=3, label="Minimum")
    plt.plot(tspan, x.max(1), 'r--', lw=3, label="Maximum")
    plt.text(1, 745, 'X(0)={}'.format(x_0), fontsize=20)

    # adding ODE solution to plot
    ode_simulator = ScipyOdeSimulator(model, tspan=tspan)
    ode_traj = ode_simulator.run()
    plt.plot(tspan, ode_traj.all[obs_name], 'g--', lw=3, label="ODE")
    plt.ylim(0, 800)
    plt.xlabel('Time')
    plt.ylabel('X molecules')

    # create histogram
    plt.subplot(122)
    plt.ylim(0, 800)
    weights = np.ones_like(x[-1, :]) / float(len(x[-1, :]))
    plt.hist(x[-1, :], 25, orientation='horizontal', weights=weights)
    plt.yticks([])
    plt.tight_layout()
    out_name = '{}.png'.format(savename)
    plt.savefig(out_name)
    plt.close()
    print("Saved {}".format(out_name))
    return out_name


def vary_x():
    """
    Creates a series of figures to demonstrate bimodal behavior of model

    """
    images = []
    for i in range(100, 400, 25):
        images.append(run(2 ** 10, i))

    try:
        import imageio
    except ImportError:
        raise ImportError("Please install imageio to create a gif")

    kargs = {'duration': 8 / len(images)}

    imageio.mimsave(
        'schloegl.gif',
        [imageio.imread(i) for i in images],
        **kargs
    )
    print("Saved schloegl.gif")


if __name__ == '__main__':
    # run(100, 100)
    vary_x()
