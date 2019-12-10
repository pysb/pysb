from pysb.examples import bngwiki_egfr_simple
from pysb.simulator import AmiciSimulator

import os


class TestScipySimulatorBase(object):
    def __init__(self):
        self.model = bngwiki_egfr_simple.model
        self.model.name = 'egfr'
        # compile the model
        self.sim = AmiciSimulator(
            model=self.model, modeldir=os.path.join(os.getcwd(),
                                                    self.model.name)
        )

    def setUp(self):
        # load the precompiled model
        self.sim = AmiciSimulator(
            model=self.model, modeldir=os.path.join(os.getcwd(),
                                                    self.model.name)
        )

    def tearDown(self):
        self.sim = None

    def test_simulation(self):
        self.sim.run()