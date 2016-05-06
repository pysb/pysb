import pysb
import os
from pysb.bng import BngConsole, generate_equations
from pysb.importers.bngl import read_bngl
import numpy


def bng_import_compare_simulations(bng_file, sim_times=range(0, 100, 10)):
    """
    This is used for manually comparing BNG file runs for the moment,
    it needs to be converted into a unit test later
    """
    m = read_bngl(bng_file)
    from pysb.generator.bng import BngGenerator
    g = BngGenerator(m)
    with open('/Users/alex/tmp/%s_pysb.bngl' %
                      os.path.basename(bng_file)[:-5], 'w') as f:
        f.write(g.get_content())

    # Simulate using the BNGL file directly
    with BngConsole(model=None) as con:
        con.load_bngl(bng_file)
        con.generate_network()
        con.action('simulate', method='ode', sample_times=sim_times)
        yfull1 = con.read_simulation_results()

    # Convert to a PySB model, then simulate using BNG
    with BngConsole(m) as con:
        con.generate_network()
        con.action('simulate', method='ode', sample_times=sim_times)
        yfull2 = con.read_simulation_results()

    # Check all species trajectories are equal (within numerical tolerance)
    for sp in range(len(yfull1) - 1):
        species = '__s%d' % sp
        print(species)
        print(yfull1[species])
        print(yfull2[species])
        print(numpy.allclose(yfull1[species], yfull2[species]))
        assert numpy.allclose(yfull1[species], yfull2[species])


def test_all_bng_files():
    bng_dir = os.path.dirname(pysb.bng._get_bng_path())
    bngl_files_dir = os.path.join(bng_dir, 'Validate')
    for f in os.listdir(bngl_files_dir):
        if f.endswith('.bngl'):
            yield (bng_import_compare_simulations, os.path.join(bngl_files_dir,
                                                                f))
