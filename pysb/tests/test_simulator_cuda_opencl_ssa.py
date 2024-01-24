import numpy as np
from nose.plugins.attrib import attr
from nose.tools import ok_
from pysb.examples.schloegl import model
from pysb.simulator import CudaSSASimulator, OpenCLSSASimulator


@attr('gpu')
class TestGpu(object):

    def setUp(self):
        self.tspan = np.linspace(0, 100, 101)
        self.model = model
        model.parameters['X_0'].value = 400
        self.simulator = CudaSSASimulator(model)
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

    def test_run_by_multi_initials_df(self):
        initials = dict()
        n_sim = 10
        for ic in self.model.initial_conditions:
            initials[ic[0]] = [ic[1].value] * n_sim
        self.simulator.run(self.tspan, initials=initials)
        ok_(self.simulator.initials.shape[0] == n_sim)

    def test_run_by_multi_params_df(self):

        n_sim = 10
        params = dict()
        for ic in self.model.parameters:
            params[ic.name] = [ic.value] * n_sim
        self.simulator.run(self.tspan, param_values=params)
        ok_(self.simulator.param_values.shape[0] == n_sim)

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


@attr('gpu')
class TestOpencl(object):

    def setUp(self):
        self.tspan = np.linspace(0, 100, 101)
        self.model = model
        model.parameters['X_0'].value = 200
        self.simulator = OpenCLSSASimulator(model)
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
