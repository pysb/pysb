import warnings
import numpy as np
from nose.plugins.attrib import attr
from pysb.examples.tyson_oscillator import model
from pysb.simulator.cupsoda import CupSodaSimulator


@attr('gpu')
def test_cupsoda_tyson():
    n_sims = 50
    vol = 1e-19
    tspan = np.linspace(0, 500, 101)
    solver = CupSodaSimulator(model, tspan=tspan, atol=1e-12, rtol=1e-12,
                           max_steps=20000, vol=vol, verbose=False)

    # Rate constants
    len_parameters = len(model.parameters)
    param_values = np.ones((n_sims, len_parameters))
    for j in range(len_parameters):
        param_values[:, j] *= model.parameters[j].value

    # Initial concentrations
    len_model_species = len(model.species)
    y0 = np.zeros((n_sims, len_model_species))

    for ic in model.initial_conditions:
        for j in range(len_model_species):
            if str(ic[0]) == str(model.species[j]):
                y0[:, j] = ic[1].value
                break

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', "Neither 'param_values' nor "
                                          "'initials' were supplied.")
        solver.run(param_values=None, initials=None)

    simres = solver.run(initials=y0)
    print(simres.observables)
    solver.run(param_values=None, initials=y0)
    solver.run(param_values=param_values, initials=y0)


@attr('gpu')
def test_memory_configs():
    n_sims = 50
    tspan = np.linspace(0, 500, 101)
    solver = CupSodaSimulator(model, tspan=tspan, atol=1e-12, rtol=1e-12,
                           max_steps=20000, verbose=False)

    # Initial concentrations
    len_model_species = len(model.species)
    y0 = np.zeros((n_sims, len_model_species))
    for ic in model.initial_conditions:
        for j in range(len_model_species):
            if str(ic[0]) == str(model.species[j]):
                y0[:, j] = ic[1].value
                break

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', "Neither 'param_values' nor "
                                          "'initials' were supplied.")
        solver.run(param_values=None, initials=None)
        
    solver.run(initials=y0) # memory_usage='sharedconstant'
    solver.opts['memory_usage'] = 'global'
    solver.run(initials=y0)
    solver.opts['memory_usage'] = 'shared'
    solver.run(initials=y0)


@attr('gpu')
def test_use_of_volume():
    n_sims = 50
    vol = 1e-19
    tspan = np.linspace(0, 500, 101)
    solver = CupSodaSimulator(model, tspan=tspan, atol=1e-12, rtol=1e-12,
                           max_steps=20000, vol=vol, verbose=False)

    # Initial concentrations
    len_model_species = len(model.species)
    y0 = np.zeros((n_sims, len_model_species))
    for ic in model.initial_conditions:
        for j in range(len_model_species):
            if str(ic[0]) == str(model.species[j]):
                y0[:, j] = ic[1].value
                break

    solver.run(initials=y0)
