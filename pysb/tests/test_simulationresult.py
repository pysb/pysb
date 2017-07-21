from pysb.simulator import ScipyOdeSimulator
from pysb.simulator.base import SimulationResult
from pysb.examples import tyson_oscillator, robertson
import numpy as np
import tempfile
from nose.tools import assert_raises
from pandas.util.testing import assert_frame_equal


def test_simres_dataframe():
    """ Test SimulationResult.dataframe() """

    tspan1 = np.linspace(0, 100, 100)
    tspan2 = np.linspace(50, 100, 50)
    tspan3 = np.linspace(100, 150, 100)
    model = tyson_oscillator.model
    sim = ScipyOdeSimulator(model, integrator='lsoda')
    simres1 = sim.run(tspan=tspan1)
    # Check retrieving a single simulation dataframe
    df_single = simres1.dataframe

    # Generate multiple trajectories
    trajectories1 = simres1.species
    trajectories2 = sim.run(tspan=tspan2).species
    trajectories3 = sim.run(tspan=tspan3).species

    # Try a simulation result with two different tspan lengths
    sim = ScipyOdeSimulator(model, param_values={'k6' : [1.,1.]}, integrator='lsoda')
    simres = SimulationResult(sim, [tspan1, tspan2], [trajectories1, trajectories2])
    df = simres.dataframe

    assert df.shape == (len(tspan1) + len(tspan2),
                        len(model.species) + len(model.observables))

    # Next try a simulation result with two identical tspan lengths, stacked
    # into a single 3D array of trajectories
    simres2 = SimulationResult(sim, [tspan1, tspan3],
                               np.stack([trajectories1, trajectories3]))
    df2 = simres2.dataframe

    assert df2.shape == (len(tspan1) + len(tspan3),
                         len(model.species) + len(model.observables))


def test_save_load():
    tspan = np.linspace(0, 100, 101)
    model = tyson_oscillator.model
    sim = ScipyOdeSimulator(model, integrator='lsoda')
    simres = sim.run(tspan=tspan, param_values={'k6': 1.0})

    sim_rob = ScipyOdeSimulator(robertson.model, integrator='lsoda')
    simres_rob = sim_rob.run(tspan=tspan)

    with tempfile.NamedTemporaryFile() as tf:
        simres.save(tf.name, dataset_name='test', append=True)

        # Try to reload when file contains only one dataset and group
        SimulationResult.load(tf.name)

        simres.save(tf.name, append=True)

        # Trying to overwrite an existing dataset gives a ValueError
        assert_raises(ValueError, simres.save, tf.name, append=True)

        # Trying to write to an existing file without append gives an IOError
        assert_raises(IOError, simres.save, tf.name)

        # Trying to write a SimulationResult to the same group with a
        # different model name results in a ValueError
        assert_raises(ValueError, simres_rob.save, tf.name,
                      dataset_name='robertson', group_name=model.name,
                      append=True)

        simres_rob.save(tf.name, append=True)

        # Trying to load from a file with more than one group without
        # specifying group_name should raise a ValueError
        assert_raises(ValueError, SimulationResult.load, tf.name)

        # Trying to load from a group with more than one dataset without
        # specifying a dataset_name should raise a ValueError
        assert_raises(ValueError, SimulationResult.load, tf.name,
                      group_name=model.name)

        # Load should succeed when specifying group_name and dataset_name
        simres_load = SimulationResult.load(tf.name, group_name=model.name,
                                            dataset_name='test')

    assert np.allclose(simres.species, simres_load.species)
    assert np.allclose(simres.tout, simres_load.tout)
    assert np.allclose(simres.param_values, simres_load.param_values)
    assert np.allclose(simres.initials, simres_load.initials)
    assert_frame_equal(simres.dataframe, simres_load.dataframe)

    assert simres.squeeze == simres_load.squeeze
    assert simres.simulator_class == simres_load.simulator_class
    assert simres.simulator_kwargs == simres_load.simulator_kwargs
    assert simres.n_sims_per_parameter_set == \
           simres_load.n_sims_per_parameter_set
    assert simres._model.name == simres_load._model.name
    assert simres.creation_date == simres_load.creation_date
    assert simres.pysb_version == simres_load.pysb_version
