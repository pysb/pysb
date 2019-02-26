from pysb.simulator.gpu_ssa import GPUSimulator
from pysb.examples.schlogl import model
import numpy as np
from nose.plugins.attrib import attr


@attr('gpu')
class TestGpu(object):

    def setUp(self):
        self.tspan = np.linspace(0, 100, 101)
        self.model = model
        model.parameters['X_0'].value = 400
        self.simulator = GPUSimulator(model)
        self.n_sim = 10

    def test_run_by_nsim(self):
        self.simulator.run(self.tspan, number_sim=self.n_sim)

    def test_run_by_multi_params(self):
        param_values = np.array(
            [p.value for p in self.model.parameters])
        param_values = np.repeat([param_values], self.n_sim, axis=0)
        self.simulator.run(self.tspan, param_values)

    def test_run_by_multi_initials(self):
        species_names = [str(s) for s in self.model.species]
        initials = np.zeros(len(species_names))
        for ic in self.model.initial_conditions:
            initials[species_names.index(str(ic[0]))] = int(ic[1].value)
        initials = np.repeat([initials], self.n_sim, axis=0)
        self.simulator.run(self.tspan, initials=initials)

    def test_run_by_params_set_n_sim(self):
        param_values = np.array(
            [p.value for p in self.model.parameters])

        self.simulator.run(self.tspan, param_values, number_sim=10)

    def test_run_by_initials_set_n_sim(self):
        species_names = [str(s) for s in self.model.species]
        initials = np.zeros(len(species_names))
        for ic in self.model.initial_conditions:
            initials[species_names.index(str(ic[0]))] = int(ic[1].value)

        self.simulator.run(self.tspan, initials=initials, number_sim=10)
