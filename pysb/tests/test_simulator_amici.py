from pysb.examples import bngwiki_egfr_simple
from pysb.simulator import AmiciSimulator

import os
import shutil


class TestScipySimulatorBase(object):
    def __init__(self):
        self.model = bngwiki_egfr_simple.model
        self.model.name = 'amici_simulator_test_bngwiki_egfr_simple'
        self.modeldir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), self.model.name
        )

        self.sim = AmiciSimulator(model=self.model, modeldir=self.modeldir)

    def __del__(self):
        if os.path.exists(self.modeldir):
            shutil.rmtree(self.modeldir)

    def setUp(self):
        # load the precompiled model
        self.sim = AmiciSimulator(model=self.model, modeldir=self.modeldir)

    def tearDown(self):
        self.sim = None

    def test_temp_compilations(self):
        self.sim = AmiciSimulator(model=self.model)

    def test_simulation_default_params(self):
        self.sim.run(tspan=[0, 1])

    def test_simulation_default_parallel(self):
        self.sim.run(
            tspan=[0, 1],
            param_values=[self.sim.param_values[0, :],
                          self.sim.param_values[0, :]]
        )
