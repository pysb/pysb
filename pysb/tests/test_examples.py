from pysb.bng import generate_network, NoInitialConditionsError, NoRulesError
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
            module = importlib.import_module(package)
            # Reset do_export to the default in case the model changed it.
            # FIXME the self-export mechanism should be more self-contained so
            # this isn't needed here.
            SelfExporter.do_export = True
            yield module.model

expected_exceptions = {
    'tutorial_b': (NoInitialConditionsError, NoRulesError),
    'tutorial_c': (NoInitialConditionsError, NoRulesError),
    }


def check_generate_network(model):
    """Tests that network generation runs without error for the given model"""
    success = False
    try:
        generate_network(model)
        success = True
    except Exception as e:
        # Some example models are deliberately incomplete, so here we will treat
        # any of these "expected" exceptions as a success.
        model_base_name = model.name.rsplit('.', 1)[1]
        exception_classes = expected_exceptions.get(model_base_name)
        if exception_classes and isinstance(e, exception_classes):
            success = True
    assert success, "Network generation failed on model %s:\n-----\n%s" % \
                (model.name, traceback.format_exc())
