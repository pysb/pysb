from pysb.examples.earm_1_0 import model
import numpy
from pysb.tools.cytoscapejs_visualization.model_visualization_reactions import ModelVisualization
from pysb.tools.cytoscapejs_visualization.util_networkx import from_networkx

t = numpy.linspace(0, 500, 100)
a = ModelVisualization(model)
data = a.dynamic_view(tspan=t, get_passengers=True)