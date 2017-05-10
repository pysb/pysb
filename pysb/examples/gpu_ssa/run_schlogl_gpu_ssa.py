import time

import numpy as np
from pysb.bng import generate_network
from pysb.integrate import odesolve

from pysb.simulator.gpu_ssa import GPUSimulator
from pysb.examples.schlogl import model
import matplotlib.pyplot as plt

tspan = np.linspace(0, 100, 101, dtype=np.float32)

name = 'X_total'

generate_network(model, verbose=False, cleanup=False, )


def plot_mean_min_max(tout, trajectories, param_values=None, title=None,
                      savename='test'):
    x = np.array([tr[:] for tr in trajectories]).T

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
    # plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("%s.png" % savename)
    plt.show()
    plt.close()


def main(number_particles, value):
    num_particles = int(number_particles)
    model.parameters['X_0'].value = value
    savename = 'test_{}'.format(int(value))
    threads = 128
    simulator = GPUSimulator(model, verbose=False, threads=threads)
    starttime = time.time()
    traj = simulator.run(tspan, number_sim=number_particles)
    end_time = time.time() - starttime
    print('Sim = %s ,Time = %s, Threads = %s\n' % (
    str(num_particles), str(end_time), threads)),
    result = np.array(traj.observables)[name]
    plot_mean_min_max(simulator.tout.T, result, savename=savename)
    return end_time


if __name__ == '__main__':
    main(2 ** 13, 100)
    quit()
    for i in range(100, 400, 100):
        main(2 ** 10, i)
