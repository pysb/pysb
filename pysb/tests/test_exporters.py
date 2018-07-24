# -*- coding:utf-8; -*-
"""
Test example models export without error using the available exporters

Based on code submitted in a PR by @keszybz in pysb/pysb#113
"""
from pysb.tests.test_examples import get_example_models, expected_exceptions
from pysb import export
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import pandas as pd
try:
    import roadrunner
except ImportError:
    roadrunner = None
from nose.plugins.skip import SkipTest


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
        if not exception_class or not isinstance(e, exception_class):
            raise

    if exported_file is not None:
        if format == 'python':
            # linspace arguments picked to avoid VODE warning
            exec(exported_file + 'Model().simulate(tspan=numpy.linspace(0,1,501))\n', {'_use_inline': False})
        elif format == 'pysb_flat':
            exec(exported_file, {'__name__': model.name})
        elif format == 'sbml':
            # Skip the simulation comparison if roadrunner not available
            if roadrunner is None:
                raise SkipTest("SBML Simulation test skipped (requires roadrunner)")

            roadrunner.Logger.setLevel(roadrunner.Logger.LOG_ERROR)

            # Simulate SBML using roadrunner
            rr = roadrunner.RoadRunner(exported_file)
            rr.timeCourseSelections = \
                ['__s{}'.format(i) for i in range(len(model.species))] + \
                ['__obs{}'.format(i) for i in range(len(model.observables))]
            rr_result = rr.simulate(0, 10, 100)

            # Simulate original using PySB
            df = ScipyOdeSimulator(model).run(tspan=np.linspace(0, 10, 100)).dataframe

            # Compare species' trajectories
            for sp_idx in range(len(model.species)):
                rr_sp = rr_result[:, sp_idx]
                py_sp = df.iloc[:, sp_idx]
                is_close = np.allclose(rr_sp, py_sp, rtol=1e-4)
                if not is_close:
                    print(pd.DataFrame(dict(rr=rr_sp, pysb=py_sp)))
                    raise ValueError('Model {}, species __s{} trajectories do not match:'.format(
                        model.name, sp_idx))

            # Compare observables' trajectories
            for obs_idx in range(len(model.observables)):
                rr_obs = rr_result[:, obs_idx + len(model.species)]
                py_obs = df.iloc[:, obs_idx + len(model.species)]
                is_close = np.allclose(rr_obs, py_obs, rtol=1e-4)
                if not is_close:
                    print(pd.DataFrame(dict(rr=rr_obs, pysb=py_obs)))
                    raise ValueError('Model {}, observable__o{} "{}" trajectories do not match:'.format(
                        model.name, obs_idx, model.observables[obs_idx].name))
