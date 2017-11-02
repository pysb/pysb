from pysb.examples.tyson_oscillator import model
import numpy
from pysb.tools.cytoscapejs_visualization.model_visualization_cytoscapejs import FluxVisualization

t = numpy.linspace(0, 100, 100)
a = FluxVisualization(model)
a.setup_info(tspan=t)
g_layout = a.dot_layout()
a.graph_to_json(layout=g_layout)
