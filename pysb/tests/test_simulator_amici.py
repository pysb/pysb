from pysb.examples import bngwiki_egfr_simple
from pysb.simulator import AmiciSimulator

import os
import shutil

try:
    import amici
except:
    amici = None


class TestAmiciSimulatorBase(object):
    def __init__(self):
        self.model = bngwiki_egfr_simple.model
        self.model.name = 'amici_simulator_test_bngwiki_egfr_simple'
        self.modeldir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), self.model.name
        )
        if amici is not None:
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
        self.sim.run(tspan=[0, 1])
        tempdir = self.sim.modeldir
        assert os.path.exists(tempdir)
        self.sim = None
        assert not os.path.exists(tempdir)

    def test_simulation_default_params(self):
        self.sim.run(tspan=[0, 1])

    def test_simulation_default_sequential_multiparam(self):
        result = self.sim.run(
            tspan=[0, 1],
            param_values=[self.sim.param_values[0, :],
                          self.sim.param_values[0, :]]
        )
        assert result.nsims == 2

    def test_simulation_default_sequential_multiinitial(self):
        result = self.sim.run(
            tspan=[0, 1],
            initials=[self.sim.initials[0, :],
                      self.sim.initials[0, :]]
        )
        assert result.nsims == 2

    def test_simulation_default_parallel_multiparam(self):
        result = self.sim.run(
            tspan=[0, 1],
            param_values=[self.sim.param_values[0, :],
                          self.sim.param_values[0, :]],
            num_processors=2
        )
        assert result.nsims == 2
