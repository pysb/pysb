import os
import pysb.pathfinder as pf
from pysb.bng import BngFileInterface
from pysb.importers.bngl import model_from_bngl, BnglImportError
from pysb.importers.sbml import model_from_sbml, model_from_biomodels
import numpy
from nose.tools import assert_raises_regexp, raises
import warnings
import mock
import tempfile
import shutil


def bngl_import_compare_simulations(bng_file, force=False,
                                    sim_times=range(0, 100, 10)):
    """
    Test BNGL file import by running an ODE simulation on the imported model
    and on the BNGL file directly to compare trajectories.
    """
    m = model_from_bngl(bng_file, force=force)

    # Simulate using the BNGL file directly
    with BngFileInterface(model=None) as bng:
        bng.action('readFile', file=bng_file, skip_actions=1)
        bng.action('generate_network')
        bng.action('simulate', method='ode', sample_times=sim_times)
        bng.execute()
        yfull1 = bng.read_simulation_results()

    # Convert to a PySB model, then simulate using BNG
    with BngFileInterface(model=m) as bng:
        bng.action('generate_network')
        bng.action('simulate', method='ode', sample_times=sim_times)
        bng.execute()
        yfull2 = bng.read_simulation_results()

    # Check all species trajectories are equal (within numerical tolerance)
    for species in range(len(yfull1)):
        for tp in range(len(yfull1[species])):
            if not numpy.isclose(yfull1[species][tp],
                                 yfull2[species][tp], atol=1e-4, rtol=1e-4):
                print(species)
                print(tp)
                print(yfull1[species][tp])
                print(yfull2[species][tp])
                raise Exception('Trajectory mismatch')


def bngl_import_compare_nfsim(bng_file):
    m = model_from_bngl(bng_file)

    BNG_SEED = 123

    # Simulate using the BNGL file directly
    with BngFileInterface(model=None) as bng:
        bng.action('readFile', file=bng_file, skip_actions=1)
        bng.action('simulate', method='nf', n_steps=10, t_end=100,
                   seed=BNG_SEED)
        bng.execute()
        yfull1 = bng.read_simulation_results()

    # Convert to a PySB model, then simulate using BNG
    with BngFileInterface(model=m) as bng:
        bng.action('simulate', method='nf', n_steps=10, t_end=100,
                   seed=BNG_SEED)
        bng.execute()
        yfull2 = bng.read_simulation_results()

    # Check all species trajectories are equal (within numerical tolerance)
    for i in range(len(m.observables)):
        print(i)
        print(yfull1[i])
        print(yfull2[i])
        print(yfull1[i] == yfull2[i])
        assert yfull1[i] == yfull2[i]


def _bng_validate_directory():
    """ Location of BNG's validation models directory"""
    bng_exec = os.path.realpath(pf.get_path('bng'))
    if bng_exec.endswith('.bat'):
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            return os.path.join(conda_prefix, 'share\\bionetgen\\Validate')

    return os.path.join(os.path.dirname(bng_exec), 'Validate')


def _bngl_location(filename):
    """
    Gets the location of one of BioNetGen's validation model files in BNG's
    Validate directory.
    """
    bngl_file = os.path.join(_bng_validate_directory(), filename + '.bngl')
    return bngl_file


def _sbml_location(filename):
    """
    Gets the location of one of BioNetGen's validation SBML files in BNG's
    Validate/INPUT_FILES directory.
    """
    sbml_file = os.path.join(
        _bng_validate_directory(), 'INPUT_FILES', filename + '.xml')
    return sbml_file


def test_bngl_import_expected_passes_with_force():
    for filename in ('continue',
                     ):
        full_filename = _bngl_location(filename)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            yield (bngl_import_compare_simulations, full_filename, True)


# TODO: Fix error in BNGXML generator that causes error with sim. below

# def test_bngl_import_expected_passes_nfsim():
#     for filename in ('isingspin_localfcn',):
#         full_filename = _bngl_location(filename)
#         yield (bngl_import_compare_nfsim, full_filename)


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
               'dupsites': 'Molecule \w* has multiple sites with the same name',
               'excludereactants': 'ListOfExcludeReactants .* not supported',
               'fixed': 'Species .* is fixed'
               }
    expected_errors = {'CaOscillate_Sat': errtype['ratelawtype'],
                       'Haugh2b': errtype['excludereactants'],
                       'Repressilator': errtype['dupsites'],
                       'Motivating_example': errtype['fixed'],
                       'Motivating_example_cBNGL': errtype['fixed'],
                       'test_synthesis_cBNGL_simple': errtype['fixed'],
                       'blbr': errtype['dupsites'],
                       'fceri_ji': errtype['dupsites'],
                       'gene_expr': errtype['fixed'],
                       'gene_expr_func': errtype['fixed'],
                       'heise': errtype['dupsites'],
                       'hybrid_test': errtype['dupsites'],
                       'isingspin_energy': errtype['ratelawmissing'],
                       'localfunc': errtype['dupsites'],
                       'test_MM': errtype['ratelawtype'],
                       'test_sat': errtype['ratelawtype'],
                       'test_fixed': errtype['dupsites'],
                       'test_paramname': errtype['dupsites'],
                       'test_synthesis_complex': errtype['fixed'],
                       'test_synthesis_complex_source_cBNGL': errtype['fixed'],
                       'test_synthesis_simple': errtype['fixed'],
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


def _sbml_for_mocks(accession_no, mirror):
    # Need to make a copy because import_from_biomodels deletes the SBML
    # after import
    _, filename = tempfile.mkstemp()
    shutil.copy(_sbml_location('test_sbml_flat_SBML'), filename)
    return filename


@mock.patch('pysb.importers.sbml._download_biomodels', _sbml_for_mocks)
def test_biomodels_import_with_mock():
    model_from_biomodels('1')


@raises(ValueError)
def test_biomodels_invalid_mirror():
    model_from_biomodels('1', mirror='spam')
