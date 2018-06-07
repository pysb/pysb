# -*- coding:utf-8; -*-
"""
Test example models export without error using the available exporters

Based on code submitted in a PR by @keszybz in pysb/pysb#113
"""
from pysb.tests.test_examples import get_example_models, expected_exceptions
from pysb import export


def test_export():
    for model in get_example_models():
        for format in export.formats:
            fn = lambda: check_convert(model, format)
            fn.description = "Check export: {} → {}".format(
                model.name, format)
            yield fn


def check_convert(model, format):
    """ Test exporters run without error """
    exported_file = None
    try:
        exported_file = export.export(model, format)
    except export.ExpressionsNotSupported:
        pass
    except export.CompartmentsNotSupported:
        pass
    except Exception as e:
        # Some example models are deliberately incomplete, so here we
        # will treat any of these "expected" exceptions as a success.
        model_base_name = model.name.rsplit('.', 1)[1]
        exception_class = expected_exceptions.get(model_base_name)
        if exception_class and isinstance(e, exception_class):
            pass
        else:
            raise

    if exported_file is not None:
        if format == 'python':
            # linspace arguments picked to avoid VODE warning
            exec(exported_file + 'Model().simulate(tspan=numpy.linspace(0,1,501))\n', {'_use_inline': False})
