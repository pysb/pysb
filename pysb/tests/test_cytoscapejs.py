from pysb.testing import *
from pysb import Monomer, Parameter, Initial, Observable, Rule, Expression
import networkx as nx
from pysb.tools.cytoscapejs_visualization.util_networkx import from_networkx
import pysb.tools.cytoscapejs_visualization.model_visualization as viz
from pysb.tools.cytoscapejs_visualization.cytoscapejs import viewer as cyjs
import numpy as np
from nose.tools import *
from pysb.simulator import SimulatorException


class CytoscapejsWidgetTests(object):

    def test_get_style_names(self):
        styles = cyjs.get_style_names()

        assert_is_not_none(styles)
        assert_equal(type(list()), type(styles))

        assert_equal(8, len(styles))

    def test_get_style(self):
        def_style = cyjs.get_style('default')

        assert_is_not_none(def_style)
        assert_equal(type(list()), type(def_style))

        print(def_style)
        assert_raises(ValueError, cyjs.get_style, 'foo')

    def test_render(self):
        g = nx.scale_free_graph(100)
        g_cyjs = from_networkx(g)
        result = cyjs.render(g_cyjs, layout_algorithm='circle')
        assert_is_none(result)


class TestModelVisualizationBase(object):
    @with_model
    def setUp(self):
        Monomer('A', ['a'])
        Monomer('B', ['b'])

        Parameter('ksynthA', 100)
        Parameter('ksynthB', 100)
        Parameter('kbindAB', 100)

        Parameter('A_init', 0)
        Parameter('B_init', 0)

        Initial(A(a=None), A_init)
        Initial(B(b=None), B_init)

        Observable("A_free", A(a=None))
        Observable("B_free", B(b=None))
        Observable("AB_complex", A(a=1) % B(b=1))

        Rule('A_synth', None >> A(a=None), ksynthA)
        Rule('B_synth', None >> B(b=None), ksynthB)
        Rule('AB_bind', A(a=None) + B(b=None) >> A(a=1) % B(b=1), kbindAB)

        self.model = model

        # Convenience shortcut for accessing model monomer objects
        self.mon = lambda m: self.model.monomers[m]

        # This timespan is chosen to be enough to trigger a Jacobian evaluation
        # on the various solvers.
        self.time = np.linspace(0, 1)
        self.viz_data = viz.ModelVisualization(self.model)

    def tearDown(self):
        self.model = None
        self.time = None
        self.sim = None


class TestModelVisualizationSingle(TestModelVisualizationBase):

    def test_static_view(self):
        self.viz_data.static_view()

    def test_dynamic_view(self):
        self.viz_data.dynamic_view(tspan=self.time)

    @raises(SimulatorException)
    def test_dynamic_view_no_time(self):
        self.viz_data.dynamic_view()

    @raises(ValueError)
    def test_non_valid_view(self):
        self.viz_data.species_graph(view='invalid')

    @raises(Exception)
    def test_non_valid_parameters(self):
        self.viz_data.dynamic_view(tspan=self.time, param_values=[1,2])

    def test_graph_nodes(self):
        graph = self.viz_data.species_graph(view='static')
        nodes = graph.nodes()
        assert len(nodes) == len(self.model.species)


