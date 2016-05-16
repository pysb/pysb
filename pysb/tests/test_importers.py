import pysb
import os
from pysb.bng import BngConsole
from pysb.importers.bngl import model_from_bngl, BnglImportError
import numpy
from nose.tools import assert_raises_regexp


def bng_import_compare_simulations(bng_file, sim_times=range(0, 100, 10)):
    """
    Test BNGL file import by running an ODE simulation on the imported model
    and on the BNGL file directly to compare trajectories.
    """
    m = model_from_bngl(bng_file)

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
        print(numpy.allclose(yfull1[species], yfull2[species]))
        assert numpy.allclose(yfull1[species], yfull2[species])


def validate_bngl_import(filename):
    bng_dir = os.path.dirname(pysb.bng._get_bng_path())
    bngl_files_dir = os.path.join(bng_dir, 'Validate')
    bng_import_compare_simulations(os.path.join(bngl_files_dir, filename))


def test_bngl_import_expected_passes():
    for filename in ('CaOscillate_Func',
                     'Haugh2b',
                     'continue',
                     'deleteMolecules',
                     'egfr_net',
                     'empty_compartments_block',
                     'gene_expr',
                     'gene_expr_func',
                     'gene_expr_simple',
                     'isomerization',
                     'michment',
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
                     'visualize'):
        yield (validate_bngl_import, filename + '.bngl')


def test_bngl_import_expected_errors():
    errtype = {'localfn': 'Function \w* is local',
               'ratelawtype': 'Rate law \w* has unknown type',
               'ratelawmissing': 'Rate law missing for rule',
               'numbonds': 'unsupported number of bonds',
               'dupsites': 'Molecule \w* has multiple sites with the same name'
              }
    expected_errors = {'ANx': errtype['localfn'],
                       'CaOscillate_Sat': errtype['ratelawtype'],
                       'Motivating_example': errtype['numbonds'],
                       'Motivating_example_cBNGL': errtype['numbonds'],
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
        yield (assert_raises_regexp,
               BnglImportError,
               errmsg,
               validate_bngl_import,
               filename + '.bngl')
