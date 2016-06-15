from pysb.tools.render_models_cytoscape import run_render_model
from pysb.examples.tyson_oscillator import model

run_render_model(model, render_type='species')