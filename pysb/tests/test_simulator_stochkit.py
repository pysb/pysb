from pysb.export.stochkit import StochKitExporter
from pysb.simulator import StochKitSimulator
from pysb.examples import robertson, earm_1_0, expression_observables
import numpy as np
from pysb.core import as_complex_pattern
from nose.tools import raises

_STOCHKIT_SEED = 123


def test_stochkit_export():
    StochKitExporter(robertson.model).export()


@raises(ValueError)
def test_stochkit_invalid_init_kwarg():
    StochKitSimulator(earm_1_0.model, tspan=range(100), spam='eggs')


def test_stochkit_earm():
    tspan = np.linspace(0, 1000, 10)
    sim = StochKitSimulator(earm_1_0.model, tspan=tspan)
    simres = sim.run(n_runs=2, seed=_STOCHKIT_SEED, algorithm="ssa")
    simres_tl = sim.run(n_runs=2, seed=_STOCHKIT_SEED, algorithm="tau_leaping")


def test_stochkit_earm_multi_initials():
    model = earm_1_0.model
    tspan = np.linspace(0, 1000, 10)
    sim = StochKitSimulator(model, tspan=tspan)
    unbound_L = model.monomers['L'](b=None)
    simres = sim.run(initials={unbound_L: [3000, 1500]},
                     n_runs=2, seed=_STOCHKIT_SEED, algorithm="ssa")
    df = simres.dataframe

    unbound_L_index = model.get_species_index(as_complex_pattern(unbound_L))

    # Check we have two repeats of each initial
    assert np.allclose(df.loc[(slice(None), 0), '__s%d' % unbound_L_index],
                       [3000, 3000, 1500, 1500])


def test_stochkit_expressions():
    model = expression_observables.model
    tspan = np.linspace(0, 100, 11)
    sim = StochKitSimulator(model, tspan=tspan)
    assert np.allclose(sim.run().tout, tspan)
