from pysb.examples import earm_1_0, schlogl
from pysb.simulator import ScipyOdeSimulator, CupSodaSimulator, \
    StochKitSimulator, GPUSimulator
import numpy as np
import timeit


class Earm10ODESuite(object):
    def setup(self):
        self.nsims = 100
        self.timer = timeit.default_timer
        self.model = earm_1_0.model
        self.parameter_set = np.ones((self.nsims, len(self.model.parameters)))
        for i in range(len(self.parameter_set)):
            for j in range(len(self.parameter_set[i])):
                self.parameter_set[i][j] *= self.model.parameters[j].value
        integrator_options_common = {
            'model': self.model,
            'tspan': np.linspace(0, 1000, 101),
            'atol': 1e-6,
            'rtol': 1e-6,
            'mxsteps': 20000,
            'param_values': self.parameter_set
        }

        self.sim_lsoda = ScipyOdeSimulator(
            integrator='lsoda',
            **integrator_options_common
        )
        self.sim_cupsoda = CupSodaSimulator(
            **integrator_options_common
        )

    def time_scipy_lsoda(self):
        self.sim_lsoda.run()

    def time_cupsoda(self):
        self.sim_cupsoda.run()


class SSASchlogl(object):
    def setup(self):
        self.nsims = 1000
        self.timer = timeit.default_timer
        self.model = schlogl.model
        self.tspan = np.linspace(0, 100, 101)
        self.stochkit_sim = StochKitSimulator(self.model, tspan=self.tspan)
        self.gpu_ssa_sim = GPUSimulator(self.model, tspan=self.tspan)

    def time_stochkit_single_thread(self):
        self.stochkit_sim.run(n_runs=self.nsims, num_processors=1)

    def time_stochkit_eight_threads(self):
        self.stochkit_sim.run(n_runs=self.nsims, num_processors=8)

    def time_gpu_ssa(self):
        self.gpu_ssa_sim.run(number_sim=self.nsims)
