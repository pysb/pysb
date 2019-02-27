import matplotlib.pyplot as plt
import numpy as np

from pysb.bng import generate_network
from pysb.examples.schlogl import model
from pysb.integrate import odesolve
from pysb.simulator.cuda_ssa import CUDASimulator

generate_network(model, verbose=False, cleanup=False, )
tspan = np.linspace(0, 100, 101, dtype=np.float32)
name = 'X_total'


def plot_mean_min_max(tout, trajectories, param_values=None, title=None,
                      savename='test'):
    x = trajectories

    if not title:
        title = name
    plt.figure()
    plt.title(title)
    plt.subplot(121)
    # plt.title('GPU-ssa')
    plt.plot(tout, x, '0.5', lw=2, alpha=0.25)  # individual trajectories
    plt.plot(tout, x.mean(1), 'k-*', lw=3, label="Mean")
    plt.plot(tout, x.min(1), 'b--', lw=3, label="Minimum")
    plt.plot(tout, x.max(1), 'r--', lw=3, label="Maximum")
    plt.text(1, 745, 'X(0)={}'.format(savename.strip('test_')), fontsize=20)
    y = odesolve(model, tspan, param_values)
    plt.plot(tout, y[name], 'g--', lw=3, label="ODE")
    plt.ylim(0, 800)
    plt.xlabel('Time')
    plt.ylabel('X molecules')

    plt.subplot(122)
    plt.ylim(0, 800)
    weights = np.ones_like(x[-1, :]) / float(len(x[-1, :]))
    plt.hist(x[-1, :], 25, orientation='horizontal', weights=weights)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("%s.png" % savename)
    plt.show()
    plt.close()


def main(number_particles, value):
    model.parameters['X_0'].value = value
    savename = 'test_{}'.format(int(value))
    simulator = CUDASimulator(model, verbose=True)

    traj = simulator.run(tspan, number_sim=number_particles)
    result = traj.dataframe[name]

    tout = result.index.levels[1].values
    result = result.unstack(0)
    result = result.as_matrix()
    plot_mean_min_max(tout, result, savename=savename)


if __name__ == '__main__':
    main(100, 100)
    quit()
    for i in range(100, 400, 100):
        main(2 ** 10, i)
