from nose.tools import *
from pysb.core import Model, SelfExporter

def with_model(func):
    """Decorate a test to set up and tear down a Model."""
    def inner(*args, **kwargs):
        model = Model(func.func_name, _export=False)
        # manually set up SelfExporter, targeting func's globals
        SelfExporter.default_model = model
        SelfExporter.target_module = func.__module__
        SelfExporter.target_globals = func.func_globals
        SelfExporter.target_globals['model'] = model
        # call the actual test function
        func(*args, **kwargs)
        # clean up the globals
        SelfExporter.cleanup()
    return make_decorator(func)(inner)
