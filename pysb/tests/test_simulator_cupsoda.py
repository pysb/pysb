import warnings
import numpy as np
from nose.plugins.attrib import attr
from pysb.examples.tyson_oscillator import model
from pysb.simulator.cupsoda import CupSodaSimulator, run_cupsoda
from nose.tools import raises
import os


@attr('gpu')
class TestCupSODASimulatorSingle(object):
    def setUp(self):
        self.n_sims = 50
        self.tspan = np.linspace(0, 500, 101)
        self.solver = CupSodaSimulator(model, tspan=self.tspan, verbose=False,
                                       integrator_options={'atol': 1e-12,
                                                           'rtol': 1e-12,
                                                           'max_steps': 20000})
        len_model_species = len(model.species)
        y0 = np.zeros((self.n_sims, len_model_species))
        for ic in model.initials:
            for j in range(len_model_species):
                if str(ic.pattern) == str(model.species[j]):
                    y0[:, j] = ic.value.value
                    break
        self.y0 = y0

    def test_use_of_volume(self):
        # Initial concentrations
        self.solver.run(initials=self.y0)
        print(self.solver.vol)
        assert self.solver.vol is None
        self.solver.vol = 1e-20
        assert self.solver.vol == 1e-20

    def test_integrator_options(self):
        assert self.solver.opts['atol'] == 1e-12
        assert self.solver.opts['rtol'] == 1e-12
        assert self.solver.opts['max_steps'] == 20000

    def test_arguments(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', "Neither 'param_values' nor "
                                              "'initials' were supplied.")
            self.solver.run(param_values=None, initials=None)

    def test_memory_usage(self):
        assert self.solver.opts['memory_usage'] == 'sharedconstant'
        self.solver.run(initials=self.y0)  # memory_usage='sharedconstant'
        self.solver.opts['memory_usage'] = 'global'
        self.solver.run(initials=self.y0)
        self.solver.opts['memory_usage'] = 'shared'
        self.solver.run(initials=self.y0)

    def test_n_blocks(self):
        print(self.solver.n_blocks)
        self.solver.n_blocks = 128
        assert self.solver.n_blocks == 128
        self.solver.run(initials=self.y0)

    def test_multi_chunks(self):
        sim = CupSodaSimulator(model, tspan=self.tspan, verbose=False,
                               initials=self.y0,
                               integrator_options={'atol': 1e-12,
                                                   'rtol': 1e-12,
                                                   'chunksize': 25,
                                                   'max_steps': 20000})
        res = sim.run()
        assert res.nsims == self.n_sims

    @raises(ValueError)
    def test_set_nblocks_str(self):
        self.solver.n_blocks = 'fail'

    @raises(ValueError)
    def test_set_nblocks_0(self):
        self.solver.n_blocks = 0

    def test_run_tyson(self):
        # Rate constants
        len_parameters = len(model.parameters)
        param_values = np.ones((self.n_sims, len_parameters))
        for j in range(len_parameters):
            param_values[:, j] *= model.parameters[j].value
        simres = self.solver.run(initials=self.y0)
        print(simres.observables)
        self.solver.run(param_values=None, initials=self.y0)
        self.solver.run(param_values=param_values, initials=self.y0)
        self.solver.run(param_values=param_values, initials=self.y0)

    def test_verbose(self):
        solver = CupSodaSimulator(model, tspan=self.tspan, verbose=True,
                                  integrator_options={'atol': 1e-12,
                                                      'rtol': 1e-12,
                                                      'vol': 1e-5,
                                                      'max_steps': 20000})
        solver.run()

    def test_run_cupsoda_instance(self):
        run_cupsoda(model, tspan=self.tspan)

    @raises(ValueError)
    def test_invalid_init_kwarg(self):
        CupSodaSimulator(model, tspan=self.tspan, spam='eggs')

    @raises(ValueError)
    def test_invalid_integrator_option(self):
        CupSodaSimulator(model, tspan=self.tspan,
                         integrator_options={'spam': 'eggs'})
