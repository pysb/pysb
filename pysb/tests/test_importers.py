import pysb
import os
from pysb.bng import BngConsole
from pysb.importers.bngl import model_from_bngl, BnglImportError
from pysb.importers.sbml import model_from_sbml
import numpy
from nose.tools import assert_raises_regexp
import warnings


def bngl_import_compare_simulations(bng_file, force=False,
                                    sim_times=range(0, 100, 10)):
    """
    Test BNGL file import by running an ODE simulation on the imported model
    and on the BNGL file directly to compare trajectories.
    """
    m = model_from_bngl(bng_file, force=force)

    # Simulate using the BNGL file directly
    with BngConsole(model=None, suppress_warnings=True) as bng:
        bng.load_bngl(bng_file)
        bng.generate_network()
        bng.action('simulate', method='ode', sample_times=sim_times)
        yfull1 = bng.read_simulation_results()

    # Convert to a PySB model, then simulate using BNG
    with BngConsole(m, suppress_warnings=True) as bng:
        bng.generate_network()
        bng.action('simulate', method='ode', sample_times=sim_times)
        yfull2 = bng.read_simulation_results()

    # Check all species trajectories are equal (within numerical tolerance)
    for species in m.species:
        print(species)
        print(yfull1[species])
        print(yfull2[species])
        print(numpy.allclose(yfull1[species], yfull2[species], atol=1e-8,
                             rtol=1e-8))
        assert numpy.allclose(yfull1[species], yfull2[species], atol=1e-8,
                              rtol=1e-8)


def _bngl_location(filename):
    """
    Gets the location of one of BioNetGen's validation model files in BNG's
    Validate directory.
    """
    bng_dir = os.path.dirname(pysb.bng._get_bng_path())
    bngl_file = os.path.join(bng_dir, 'Validate', filename + '.bngl')
    return bngl_file


def _sbml_location(filename):
    """
    Gets the location of one of BioNetGen's validation SBML files in BNG's
    Validate/INPUT_FILES directory.
    """
    bng_dir = os.path.dirname(pysb.bng._get_bng_path())
    sbml_file = os.path.join(bng_dir, 'Validate/INPUT_FILES', filename +
                             '.xml')
    return sbml_file


def test_bngl_import_expected_passes_with_force():
    for filename in ('Haugh2b',
                     'continue',
                     'gene_expr',
                     'gene_expr_func',
                     'Motivating_example',
                     'Motivating_example_cBNGL',
                     'test_synthesis_cBNGL_simple',
                     'test_synthesis_complex',
                     'test_synthesis_complex_source_cBNGL',
                     'test_synthesis_simple'
                     ):
        full_filename = _bngl_location(filename)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            yield (bngl_import_compare_simulations, full_filename, True)


def test_bngl_import_expected_passes():
    for filename in ('CaOscillate_Func',
                     'deleteMolecules',
                     'egfr_net',
                     'empty_compartments_block',
                     'gene_expr_simple',
                     'isomerization',
                     'michment',
                     'motor',
                     'simple_system',
                     'test_compartment_XML',
                     'test_setconc',
                     'test_synthesis_complex_0_cBNGL',
                     'toy-jim',
                     'univ_synth',
                     'visualize'):
        full_filename = _bngl_location(filename)
        yield (bngl_import_compare_simulations, full_filename)


def test_bngl_import_expected_errors():
    errtype = {'localfn': 'Function \w* is local',
               'ratelawtype': 'Rate law \w* has unknown type',
               'ratelawmissing': 'Rate law missing for rule',
               'dupsites': 'Molecule \w* has multiple sites with the same name'
              }
    expected_errors = {'ANx': errtype['localfn'],
                       'CaOscillate_Sat': errtype['ratelawtype'],
                       'Repressilator': errtype['dupsites'],
                       'blbr': errtype['dupsites'],
                       'fceri_ji': errtype['dupsites'],
                       'heise': errtype['dupsites'],
                       'hybrid_test': errtype['dupsites'],
                       'isingspin_energy': errtype['ratelawmissing'],
                       'isingspin_localfcn': errtype['localfn'],
                       'localfunc': errtype['dupsites'],
                       'test_MM': errtype['ratelawtype'],
                       'test_sat': errtype['ratelawtype'],
                       'test_fixed': errtype['dupsites'],
                       'test_paramname': errtype['dupsites'],
                       'tlbr': errtype['dupsites'],
                       'tlmr': errtype['dupsites']
                       }

    for filename, errmsg in expected_errors.items():
        full_filename = _bngl_location(filename)
        yield (assert_raises_regexp,
               BnglImportError,
               errmsg,
               bngl_import_compare_simulations,
               full_filename)


def test_sbml_import_flat_model():
    model_from_sbml(_sbml_location('test_sbml_flat_SBML'))


def test_sbml_import_structured_model():
    model_from_sbml(_sbml_location('test_sbml_structured_SBML'), atomize=True)
