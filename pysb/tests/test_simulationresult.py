from pysb.simulator import ScipyOdeSimulator, BngSimulator
from pysb.simulator.base import SimulationResult
from pysb.examples import tyson_oscillator, robertson, \
    expression_observables, earm_1_3, bax_pore_sequential, bax_pore, \
    bngwiki_egfr_simple
from pysb.bng import generate_equations
import numpy as np
import tempfile
from nose.tools import assert_raises, raises
import warnings
from pysb.pattern import SpeciesPatternMatcher
import collections
import copy
import io
import pandas as pd


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


def test_simres_observable():
    """ Test on demand observable evaluation """
    models = [tyson_oscillator.model, robertson.model,
              expression_observables.model, earm_1_3.model,
              bax_pore_sequential.model, bax_pore.model,
              bngwiki_egfr_simple.model]
    for model in models:
        generate_equations(model)
        spm = SpeciesPatternMatcher(model)
        for obs in model.observables:
            dyn_obs = spm.match(pattern=obs.reaction_pattern, index=True,
                                counts=True)

            # Need to sort by species numerical order for comparison purposes
            dyn_obs = collections.OrderedDict(sorted(dyn_obs.items()))

            dyn_species = list(dyn_obs.keys())

            if obs.match == 'species':
                dyn_coeffs = [1] * len(dyn_obs)
            else:
                dyn_coeffs = list(dyn_obs.values())

            assert dyn_species == obs.species
            assert dyn_coeffs == obs.coefficients


class TestSimulationResultEarm13(object):
    def setUp(self):
        self.model = earm_1_3.model
        self.tspan = np.linspace(0, 100, 101)
        self.sim = ScipyOdeSimulator(self.model, tspan=self.tspan)
        self.simres = self.sim.run()

    @raises(ValueError)
    def test_dynamic_observable_nonpattern(self):
        self.simres.observable('cSmac')

    @raises(ValueError)
    def test_match_nonexistent_pattern(self):
        m = self.model.monomers
        self.simres.observable(m.cSmac() % m.Bid())

    def test_on_demand_observable(self):
        m = self.model.monomers
        assert isinstance(self.simres.observable(m.cSmac()), pd.Series)


def test_save_load():
    tspan = np.linspace(0, 100, 101)
    # Make a copy of model so other tests etc. don't see the changed name.
    model = copy.deepcopy(tyson_oscillator.model)
    test_unicode_name = u'Hello \u2603 and \U0001f4a9!'
    model.name = test_unicode_name
    sim = ScipyOdeSimulator(model, integrator='lsoda')
    simres = sim.run(tspan=tspan, param_values={'k6': 1.0})

    sim_rob = ScipyOdeSimulator(robertson.model, integrator='lsoda')
    simres_rob = sim_rob.run(tspan=tspan)

    # Reset equations from any previous network generation
    robertson.model.reset_equations()
    A = robertson.model.monomers['A']

    # NFsim without expressions
    nfsim1 = BngSimulator(robertson.model)
    nfres1 = nfsim1.run(n_runs=2, method='nf', tspan=np.linspace(0, 1))
    # Test attribute saving (text, float, list)
    nfres1.custom_attrs['note'] = 'NFsim without expressions'
    nfres1.custom_attrs['pi'] = 3.14
    nfres1.custom_attrs['some_list'] = [1, 2, 3]

    # NFsim with expressions
    nfsim2 = BngSimulator(expression_observables.model)
    nfres2 = nfsim2.run(n_runs=1, method='nf', tspan=np.linspace(0, 100, 11))

    with tempfile.NamedTemporaryFile() as tf:
        # Cannot have two file handles on Windows
        tf.close()

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
        assert simres_load._model.name == test_unicode_name

        # Saving network free results requires include_obs_exprs=True,
        # otherwise a warning should be raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", UserWarning)
            nfres1.save(tf.name, dataset_name='nfsim_no_obs', append=True)
            assert len(w) == 1

        nfres1.save(tf.name, include_obs_exprs=True,
                    dataset_name='nfsim test', append=True)

        # NFsim load
        nfres1_load = SimulationResult.load(tf.name,
                                            group_name=nfres1._model.name,
                                            dataset_name='nfsim test')

        # NFsim with expression
        nfres2.save(tf.name, include_obs_exprs=True, append=True)
        nfres2_load = SimulationResult.load(tf.name,
                                            group_name=nfres2._model.name)

    _check_resultsets_equal(simres, simres_load)
    _check_resultsets_equal(nfres1, nfres1_load)
    _check_resultsets_equal(nfres2, nfres2_load)


def test_save_load_observables_expressions():
    buff = io.BytesIO()
    tspan = np.linspace(0, 100, 100)
    sim = ScipyOdeSimulator(tyson_oscillator.model, tspan).run()
    sim.save(buff, include_obs_exprs=True)

    sim2 = SimulationResult.load(buff)
    assert len(sim2.observables) == len(tspan)
    # Tyson oscillator doesn't have expressions
    assert_raises(ValueError, lambda: sim2.expressions)


def _check_resultsets_equal(res1, res2):
    try:
        assert np.allclose(res1.species, res2.species)
    except ValueError:
        # Network free simulations don't have species
        pass
    assert np.allclose(res1.tout, res2.tout)
    assert np.allclose(res1.param_values, res2.param_values)
    
    if isinstance(res1.initials, np.ndarray):
        assert np.allclose(res1.initials, res2.initials)
    else:
        for k, v in res1.initials.items():
            assert np.allclose(res1.initials[k], v)

    assert np.allclose(res1._yobs_view, res2._yobs_view)
    if res1._model.expressions_dynamic():
        assert np.allclose(res1._yexpr_view, res2._yexpr_view)

    assert res1.squeeze == res2.squeeze
    assert res1.simulator_class == res2.simulator_class
    assert res1.init_kwargs == res2.init_kwargs
    assert res1.run_kwargs == res2.run_kwargs
    assert res1.n_sims_per_parameter_set == \
           res2.n_sims_per_parameter_set
    assert res1._model.name == res2._model.name
    assert res1.timestamp == res2.timestamp
    assert res1.pysb_version == res2.pysb_version
    assert res1.custom_attrs == res2.custom_attrs
