from pysb.simulator import ScipyOdeSimulator
from pysb.simulator.base import SimulationResult
from pysb.examples import tyson_oscillator
import numpy as np

def test_simres_dataframe():
    """ Test SimulationResult.dataframe() """

    # We don't currently have a Simulator which does multiple simulations at
    # a time, so we fake it to test SimulationResult.dataframe()
    tspan1 = np.linspace(0, 100, 100)
    tspan2 = np.linspace(50, 100, 50)
    model = tyson_oscillator.model
    sim = ScipyOdeSimulator(model, integrator='lsoda')
    trajectories1 = sim.run(tspan=tspan1).species
    trajectories2 = sim.run(tspan=tspan2).species
    sim.tout = [tspan1, tspan2]
    df = SimulationResult(sim, [trajectories1, trajectories2]).dataframe

    assert df.shape == (len(tspan1) + len(tspan2),
                        len(model.species) + len(model.observables))
