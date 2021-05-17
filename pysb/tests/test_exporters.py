# -*- coding:utf-8; -*-
"""
Test example models export without error using the available exporters

Based on code submitted in a PR by @keszybz in pysb/pysb#113
"""
from pysb.tests.test_examples import get_example_models, expected_exceptions
from pysb import export
from pysb.export.json import JsonExporter
from pysb.simulator import ScipyOdeSimulator
from pysb.importers.bngl import model_from_bngl
import numpy as np
import pandas as pd
import tempfile
import os
import sys
try:
    import roadrunner
except ImportError:
    roadrunner = None
from nose.plugins.skip import SkipTest
from pysb.importers.json import model_from_json
from pysb.testing import check_model_against_component_list


# Pairs of model, format that are expected to be incompatible.
skip_combinations = {
    ('fixed_initial', 'kappa'),
}


def test_export():
    for model in get_example_models():
        for format in export.formats:
            if (base_name(model), format) in skip_combinations:
                continue
            fn = lambda: check_convert(model, format)
            fn.description = "Check export: {} → {}".format(
                model.name, format)
            yield fn


def base_name(model):
    # This needs to handle names with zero or more dots.
    return model.name.split('.')[-1]


def check_convert(model, format):
    """ Test exporters run without error """
    exported_file = None
    try:
        if format == 'json':
            exported_file = JsonExporter(model).export(include_netgen=True)
        else:
            exported_file = export.export(model, format)
    except export.ExpressionsNotSupported:
        pass
    except export.CompartmentsNotSupported:
        pass
    except export.LocalFunctionsNotSupported:
        pass
    except Exception as e:
        # Some example models are deliberately incomplete, so here we
        # will treat any of these "expected" exceptions as a success.
        exception_class = expected_exceptions.get(base_name(model))
        if not exception_class or not isinstance(e, exception_class):
            raise

    if exported_file is not None:
        if format == 'pysb_flat':
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
        elif format == 'json':
            # Round-trip the model by re-importing the JSON
            m = model_from_json(exported_file)
            # Check network generation and force RHS evaluation
            if model.name not in ('pysb.examples.tutorial_b',
                                  'pysb.examples.tutorial_c'):
                ScipyOdeSimulator(m, compiler='cython')
                if sys.version_info.major >= 3:
                    # Only check on Python 3 to avoid string-to-unicode encoding
                    # issues
                    check_model_against_component_list(
                        m, model.all_components())
                # Check observable generation
                for obs in model.observables:
                    assert obs.coefficients == \
                        m.observables[obs.name].coefficients
                    assert obs.species == \
                        m.observables[obs.name].species
        elif format == 'bngl':
            if model.name.endswith('tutorial_b') or \
                    model.name.endswith('tutorial_c'):
                # Models have no rules
                return
            with tempfile.NamedTemporaryFile(suffix='.bngl',
                                             delete=False) as tf:
                tf.write(exported_file.encode('utf8'))
                # Cannot have two simultaneous file handled on Windows
                tf.close()

                try:
                    m = model_from_bngl(tf.name)
                    # Generate network and force RHS evaluation
                    ScipyOdeSimulator(m, compiler='cython')
                finally:
                    os.unlink(tf.name)
