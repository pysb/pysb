from pysb.examples.tyson_oscillator import model
from pysb.cupsoda import CupsodaSolver
import numpy as np
import warnings
from nose.plugins.attrib import attr


@attr('gpu')
def test_cupsoda_tyson():
    tspan = np.linspace(0, 500, 501)

    solver = CupsodaSolver(model, tspan=tspan, atol=1e-12, rtol=1e-6,
                           max_steps=20000, verbose=False)

    n_sims = 3

    # Rate constants
    param_values = np.ones((n_sims, len(model.parameters)))
    for i in range(len(param_values)):
        for j in range(len(param_values[i])):
            param_values[i][j] *= model.parameters[j].value

    # Initial concentrations
    y0 = np.zeros((n_sims, len(model.species)))
    for i in range(len(y0)):
        for ic in model.initial_conditions:
            for j in range(len(y0[i])):
                if str(ic[0]) == str(model.species[j]):
                    y0[i][j] = ic[1].value
                    break

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', "Neither 'y0' nor 'param_values' "
                                          "were supplied.")
        solver.run(param_values=None, y0=None)

    solver.run(param_values=param_values, y0=None)
    solver.run(param_values=None, y0=y0)
    solver.run(param_values=param_values, y0=y0)
