from pysb.bng import generate_network
from pysb.core import SelfExporter
import traceback
import os
import importlib

def test_generate_network():
    """Run network generation on all example models"""
    for model in get_example_models():
        yield (check_generate_network, model)

def get_example_models():
    """Generator that yields the model objects for all example models"""
    example_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    for filename in os.listdir(example_dir):
        if filename.endswith('.py') and not filename.startswith('run_') \
               and not filename.startswith('__'):
            modelname = filename[:-3]  # strip .py
            package = 'pysb.examples.' + modelname
            # FIXME the self-export mechanism should be more self-contained so
            # this isn't needed here.
            SelfExporter.do_export = True
            module = importlib.import_module(package)
            yield module.model

def check_generate_network(model):
    """Tests that network generation runs without error for the given model"""
    success = False
    try:
        generate_network(model)
        success = True
    except:
        pass
    assert success, "Network generation failed on model %s:\n-----\n%s" % \
                (model.name, traceback.format_exc())
