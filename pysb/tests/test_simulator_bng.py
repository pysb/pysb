from pysb.testing import *
import numpy as np
from pysb import Monomer, Parameter, Initial, Observable, Rule
from pysb.simulator.bng import BngSimulator, PopulationMap
from pysb.bng import generate_equations
from pysb.examples import robertson, expression_observables, earm_1_0

_BNG_SEED = 123


class TestBngSimulator(object):
    @with_model
    def setUp(self):
        Monomer('A', ['a'])
        Monomer('B', ['b'])

        Parameter('ksynthA', 100)
        Parameter('ksynthB', 100)
        Parameter('kbindAB', 100)

        Parameter('A_init', 0)
        Parameter('B_init', 0)

        Initial(A(a=None), A_init)
        Initial(B(b=None), B_init)

        Observable("A_free", A(a=None))
        Observable("B_free", B(b=None))
        Observable("AB_complex", A(a=1) % B(b=1))

        Rule('A_synth', None >> A(a=None), ksynthA)
        Rule('B_synth', None >> B(b=None), ksynthB)
        Rule('AB_bind', A(a=None) + B(b=None) >> A(a=1) % B(b=1), kbindAB)

        self.model = model
        generate_equations(self.model)

        # Convenience shortcut for accessing model monomer objects
        self.mon = lambda m: self.model.monomers[m]

        # This timespan is chosen to be enough to trigger a Jacobian evaluation
        # on the various solvers.
        self.time = np.linspace(0, 1)
        self.sim = BngSimulator(self.model, tspan=self.time)

    def test_1_simulation(self):
        x = self.sim.run()
        assert np.allclose(x.tout, self.time)

    def test_multi_simulations(self):
        x = self.sim.run(n_runs=10)
        assert np.shape(x.observables) == (10, 50)
        # Check initials are getting correctly reset on each simulation
        assert np.allclose(x.species[-1][0, :], x.species[0][0, :])

    def test_change_parameters(self):
        x = self.sim.run(n_runs=10, param_values={'ksynthA': 200},
                         initials={self.model.species[0]: 100})
        species = np.array(x.all)
        assert species[0][0][0] == 100.

    def test_bng_pla(self):
        self.sim.run(n_runs=5, method='pla', seed=_BNG_SEED)

    def test_tout_matches_tspan(self):
        # Linearly spaced, starting from 0
        assert all(self.sim.run(tspan=[0, 10, 20]).tout[0] == [0, 10, 20])
        # Non-linearly spaced, starting from 0
        assert all(self.sim.run(tspan=[0, 10, 30]).tout[0] == [0, 10, 30])
        # Linearly spaced, starting higher than 0
        assert all(self.sim.run(tspan=[10, 20, 30]).tout[0] == [10, 20, 30])
        # Linearly spaced, starting higher than 0
        assert all(self.sim.run(tspan=[5, 20, 30]).tout[0] == [5, 20, 30])

    def tearDown(self):
        self.model = None
        self.time = None
        self.sim = None


def test_bng_ode_with_expressions():
    model = expression_observables.model
    model.reset_equations()

    sim = BngSimulator(model, tspan=np.linspace(0, 1))
    x = sim.run(n_runs=1, method='ode')
    assert len(x.expressions) == 50
    assert len(x.observables) == 50


def test_nfsim():
    model = robertson.model
    # Reset equations from any previous network generation
    model.reset_equations()

    sim = BngSimulator(model, tspan=np.linspace(0, 1))
    x = sim.run(n_runs=1, method='nf', seed=_BNG_SEED)
    observables = np.array(x.observables)
    assert len(observables) == 50

    A = model.monomers['A']
    x = sim.run(n_runs=2, method='nf', tspan=np.linspace(0, 1),
                initials={A(): 100}, seed=_BNG_SEED)
    assert np.allclose(x.dataframe.loc[0, 0.0], [100.0, 0.0, 0.0])


def test_hpp():
    model = robertson.model
    # Reset equations from any previous network generation
    model.reset_equations()

    A = robertson.model.monomers['A']
    klump = Parameter('klump', 10000, _export=False)
    model.add_component(klump)

    population_maps = [
        PopulationMap(A(), klump)
    ]

    sim = BngSimulator(model, tspan=np.linspace(0, 1))
    x = sim.run(n_runs=1, method='nf', population_maps=population_maps,
                seed=_BNG_SEED)
    observables = np.array(x.observables)
    assert len(observables) == 50
