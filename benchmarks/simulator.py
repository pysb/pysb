from pysb.examples import earm_1_0
from pysb.simulator import ScipyOdeSimulator, CupSodaSimulator
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
