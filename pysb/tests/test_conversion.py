# -*- coding:utf-8; -*-
import os.path
import glob

from pysb.tests.test_examples import get_example_models, expected_exceptions
from pysb import export

def test_conversion():
    for model in get_example_models():
        for format in export.formats:
            fn = lambda: check_convert(model, format)
            fn.description = "Check conversion: {} → {}".format(model.name, format)
            yield fn

def check_convert(model, format):
    """Tests that conversion runs without error for the given model and format"""
    try:
        export.export(model, format)
    except Exception as e:
        # Some example models are deliberately incomplete, so here we will treat
        # any of these "expected" exceptions as a success.
        model_base_name = model.name.rsplit('.', 1)[1]
        exception_class = expected_exceptions.get(model_base_name)
        if exception_class and isinstance(e, exception_class):
            pass
        else:
            raise
