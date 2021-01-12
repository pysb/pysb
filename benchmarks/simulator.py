from pysb.examples import earm_1_0
from . import egfr_extended
from pysb.simulator import ScipyOdeSimulator, CupSodaSimulator
from pysb.bng import generate_equations
import numpy as np
import timeit


class Earm10ODESuite(object):
    def setup(self):
        self.nsims = 100
        self.timer = timeit.default_timer
        self.model = earm_1_0.model
        self.model.reset_equations()
        self.parameter_set = np.repeat(
            [[p.value for p in self.model.parameters]], self.nsims, axis=0)
        integrator_options_common = {
            'model': self.model,
            'tspan': np.linspace(0, 20000, 101),
            'param_values': self.parameter_set
        }

        self.sim_lsoda = ScipyOdeSimulator(
            integrator='lsoda',
            compiler='cython',
            integrator_options={'atol': 1e-6, 'rtol': 1e-6, 'mxstep': 20000},
            **integrator_options_common
        )

        self.sim_cupsoda = CupSodaSimulator(
            integrator_options={'atol': 1e-6, 'rtol': 1e-6, 'max_steps': 20000},
            **integrator_options_common
        )

    def time_scipy_lsoda(self):
        self.sim_lsoda.run()

    def time_cupsoda(self):
        self.sim_cupsoda.run()


class EgfrExtendedCodegenSuite(object):

    def setup_cache(self):
        model = egfr_extended.model
        model.reset_equations()
        generate_equations(model, max_iter=6)
        return model

    def time_init_python(self, model):
        self.sim_python = ScipyOdeSimulator(
            compiler='python',
            model=model,
        )

    def time_init_cython(self, model):
        self.sim_cython = ScipyOdeSimulator(
            compiler='cython',
            model=model,
        )


class EgfrExtendedRunSuite(object):

    def setup_cache(self):
        model = egfr_extended.model
        model.reset_equations()
        generate_equations(model, max_iter=6)
        common_options = {
            'model': model,
            'tspan': np.linspace(0, 10000, 1000),
            'integrator': 'vode',
            'cleanup': False,
        }
        compilers = 'python', 'cython'
        return {
            c: ScipyOdeSimulator(compiler=c, **common_options)
            for c in compilers
        }

    def time_run_python(self, sims):
        sims['python'].run()

    def time_run_cython(self, sims):
        sims['cython'].run()
