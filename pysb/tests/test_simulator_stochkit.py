from pysb.export.stochkit import StochKitExporter
from pysb.simulator import StochKitSimulator
from pysb.examples import robertson, earm_1_0
import numpy as np

_STOCHKIT_SEED = 123


def test_stochkit_export():
    StochKitExporter(robertson.model).export()


def test_stochkit_earm():
    tspan = np.linspace(0, 1000, 10)
    sim = StochKitSimulator(earm_1_0.model, tspan=tspan)
    simres = sim.run(n_runs=2, seed=_STOCHKIT_SEED, algorithm="ssa")
