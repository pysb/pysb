from pysb.testing import *
from pysb.examples.earm_1_0 import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator
from pysb.tools.model_visualization import FluxVisualization
from pysb.simulator.base import SimulatorException


class TestFluxVisualization(object):
    def setUp(self):
        self.model = model
        self.time = np.linspace(0, 20000, 100)
        self.sim = ScipyOdeSimulator.execute(self.model, tspan=self.time)

    def tearDown(self):
        self.model = None
        self.time = None
        self.sim = None

    def test_species_rendering(self):
        """Test species rendering"""
        visualization_species = FluxVisualization(self.model)
        visualization_species.visualize(render_type='species')

    def test_reactions_rendering(self):
        """Test reactions rendering"""
        visualization_reactions = FluxVisualization(self.model)
        visualization_reactions.visualize(render_type='reactions')

    def test_flux_visualization(self):
        """Test flux visualization"""
        visualization_flux = FluxVisualization(self.model)
        visualization_flux.visualize(tspan=self.time, render_type='flux')

    def test_record_video(self):
        """Test record video option"""
        visualization_video = FluxVisualization(self.model)
        visualization_video.visualize(tspan=self.time, render_type='flux', save_video=True)

    @raises(ValueError)
    def test_invalid_rendering_option(self):
        """Test invalid rendering option"""
        visualization_invalid = FluxVisualization(self.model)
        visualization_invalid.visualize(render_type='hello')

    @raises(SimulatorException)
    def test_time_not_defined(self):
        """Test time not defined"""
        visualization_time = FluxVisualization(self.model)
        visualization_time.visualize(tspan=None, render_type='flux')

    @raises(Exception)
    def test_different_parameters_length(self):
        """Test parameters entered are same length"""
        visualization_pars = FluxVisualization(self.model)
        visualization_pars.visualize(tspan=self.time, param_values=[1, 2, 3], render_type='flux',)





