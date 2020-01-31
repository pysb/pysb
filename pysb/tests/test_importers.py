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
from pysb.logging import get_logger

# Some models don't match BNG originals exactly due to loss of numerical
# precision. See https://github.com/pysb/pysb/issues/443
REDUCED_PRECISION = {
    'CaOscillate_Func': 1e-4,
    'michment': 1e-8,
    'motor': 1e-8,
    'Repressilator': 1e-11,
}

logger = get_logger(__name__)


def bngl_import_compare_simulations(bng_file, force=False,
                                    precision=1e-12,
                                    sim_times=range(0, 100, 10)):
    """
    Test BNGL file import by running an ODE simulation on the imported model
    and on the BNGL file directly to compare trajectories.
    """
    m = model_from_bngl(bng_file, force=force)

    if sim_times is None:
        # Skip simulation check
        return

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

    # Don't check trajectories on forced examples
    if force:
        return

    assert len(yfull1.dtype.names) == len(yfull2.dtype.names)
    for species in yfull1.dtype.names:
        logger.debug(species)
        logger.debug(yfull1[species])
        if species in yfull2.dtype.names:
            renamed_species = species
        else:
            renamed_species = 'Obs_{}'.format(species)
        logger.debug(yfull2[renamed_species])
        assert numpy.allclose(yfull1[species], yfull2[renamed_species],
                              atol=precision, rtol=precision)


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
    for filename in ('Haugh2b',
                     'Motivating_example',
                     ):
        full_filename = _bngl_location(filename)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            yield (bngl_import_compare_simulations, full_filename, True)


def test_bngl_import_expected_passes_nfsim():
    for filename in ('isingspin_localfcn', ):
        full_filename = _bngl_location(filename)
        yield (bngl_import_compare_nfsim, full_filename)


def test_bngl_import_expected_passes_no_sim():
    """ These models convert properly, but we cannot generate network """
    for filename in ('blbr',         # Uses max_stoich option for netgen
                     'hybrid_test',  # Population maps are not converted
                     'tlbr'):        # Uses max_iter option for netgen
        full_filename = _bngl_location(filename)
        yield (bngl_import_compare_simulations, full_filename, False, None,
               None)


def test_bngl_import_expected_passes():
    for filename in ('CaOscillate_Func',
                     'continue',
                     'deleteMolecules',
                     'egfr_net',
                     'empty_compartments_block',
                     'gene_expr',
                     'gene_expr_func',
                     'gene_expr_simple',
                     'isomerization',
                     'localfunc',
                     'michment',
                     'Motivating_example_cBNGL',
                     'motor',
                     'simple_system',
                     'test_compartment_XML',
                     'test_setconc',
                     'test_synthesis_cBNGL_simple',
                     'test_synthesis_complex',
                     'test_synthesis_complex_0_cBNGL',
                     'test_synthesis_complex_source_cBNGL',
                     'test_synthesis_simple',
                     'toy-jim',
                     'univ_synth',
                     'visualize',
                     'Repressilator',
                     'fceri_ji',
                     'test_paramname',
                     'tlmr'):
        full_filename = _bngl_location(filename)
        yield (bngl_import_compare_simulations, full_filename, False,
               REDUCED_PRECISION.get(filename, 1e-12))


def test_bngl_import_expected_errors():
    errtype = {'ratelawtype': 'Rate law \w* has unknown type',
               'ratelawmissing': 'Rate law missing for rule',
               'plusminus': 'PLUS/MINUS state values',
               'statelabels': 'BioNetGen component/state labels are not yet supported',
              }
    expected_errors = {'ANx': errtype['plusminus'],
                       'CaOscillate_Sat': errtype['ratelawtype'],
                       'heise': errtype['statelabels'],
                       'isingspin_energy': errtype['ratelawmissing'],
                       'test_MM': errtype['ratelawtype'],
                       'test_sat': errtype['ratelawtype'],
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
