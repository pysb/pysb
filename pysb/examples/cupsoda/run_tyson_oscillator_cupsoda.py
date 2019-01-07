from pysb.examples.tyson_oscillator import model
from pysb.simulator.cupsoda import CupSodaSimulator
import numpy as np
import matplotlib.pyplot as plt


def run():
    n_sims = 100
    vol = model.parameters['vol'].value
    tspan = np.linspace(0, 500, 501)
    sim = CupSodaSimulator(model, tspan, verbose=True,
                           integrator_options={'atol' : 1e-12,
                                               'rtol' : 1e-6,
                                               'vol': vol,
                                               'max_steps' :20000})

    # Rate constants
    param_values = np.ones((n_sims, len(model.parameters)))
    for i in range(len(param_values)):
        for j in range(len(param_values[i])):
            param_values[i][j] *= model.parameters[j].value

    # Initial concentrations
    initials = np.zeros((n_sims, len(model.species)))
    for i in range(len(initials)):
        for ic in model.initials:
            for j in range(len(initials[i])):
                if str(ic.pattern) == str(model.species[j]):
                    initials[i][j] = ic.value.value
                    break

    x = sim.run(initials=initials, param_values=param_values)

    # Plot results of the first simulation
    t = x.tout[0]
    plt.plot(t, x.all[0]['CT'], lw=2, label='CT')  # should be constant
    plt.plot(t, x.all[0]['YT'], lw=2, label='YT')
    plt.plot(t, x.all[0]['M'],  lw=2, label='M')

    plt.xlabel('time')
    plt.ylabel('population')
    plt.legend(loc=0)

    plt.show()


if __name__ == '__main__':
    run()
