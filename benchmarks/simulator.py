from pysb.examples import earm_1_0, schloegl
from pysb.simulator import ScipyOdeSimulator, CupSodaSimulator, \
    StochKitSimulator, CudaSSASimulator, OpenCLSSASimulator
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
        self.sim_lsoda_no_compiler_directives = ScipyOdeSimulator(
            integrator='lsoda',
            compiler='cython',
            cython_directives={},
            integrator_options={'atol': 1e-6, 'rtol': 1e-6, 'mxstep': 20000},
            **integrator_options_common
        )

        self.sim_cupsoda = CupSodaSimulator(
            integrator_options={'atol': 1e-6, 'rtol': 1e-6, 'max_steps': 20000},
            **integrator_options_common
        )

    def time_scipy_lsoda(self):
        self.sim_lsoda.run()

    def time_scipy_lsoda_no_compiler_directives(self):
        self.sim_lsoda_no_compiler_directives.run()

    def time_cupsoda(self):
        self.sim_cupsoda.run()


class SSASchloegl(object):
    def setup_cache(self):
        self.nsims = 100
        self.timer = timeit.default_timer
        self.model = schloegl.model
        self.tspan = np.linspace(0, 100, 101)
        self.stochkit_sim = StochKitSimulator(self.model, tspan=self.tspan)
        self.cuda_ssa_sim = CudaSSASimulator(self.model, tspan=self.tspan)
        self.opencl_ssa_sim = OpenCLSSASimulator(self.model, tspan=self.tspan)

    def time_stochkit_single_thread(self):
        self.stochkit_sim.run(n_runs=self.nsims, num_processors=1)

    def time_stochkit_eight_threads(self):
        self.stochkit_sim.run(n_runs=self.nsims, num_processors=8)

    def time_gpu_ssa(self):
        self.cuda_ssa_sim.run(number_sim=self.nsims)

    def time_opencl_ssa(self):
        self.opencl_ssa_sim.run(number_sim=self.nsims)
