import os

import numpy

from earm.lopez_embedded import model
from tropical.helper_functions import read_pars
from tropical.visualization.model_visualization_reactions import ModelVisualization

directory = os.path.dirname(__file__)
parameter_path = os.path.join(directory, "pars_embedded_26.txt")
parameters = read_pars(parameter_path)
t = numpy.linspace(0, 20000, 100)
a = ModelVisualization(model)
a._setup_dynamics(tspan=t, param_values=parameters, get_passengers=True)
g_layout = a.dot_layout()
# a.graph_to_json(layout=g_layout)
