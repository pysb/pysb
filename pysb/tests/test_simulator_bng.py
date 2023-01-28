from pysb.testing import *
import numpy as np
from pysb import Monomer, Parameter, Initial, Observable, Rule, Expression
from pysb.simulator.bng import BngSimulator, PopulationMap
from pysb.bng import generate_equations
from pysb.examples import robertson, expression_observables

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


class TestNfSim(object):
    def setUp(self):
        self.model = robertson.model
        self.model.reset_equations()
        self.sim = BngSimulator(self.model, tspan=np.linspace(0, 1))
        self.param_values_2sets = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]

    def test_nfsim_2runs(self):
        x = self.sim.run(n_runs=1, method='nf', seed=_BNG_SEED)
        observables = np.array(x.observables)
        assert len(observables) == 50

        A = self.model.monomers['A']
        x = self.sim.run(n_runs=2, method='nf', tspan=np.linspace(0, 1),
                    initials={A(): 100}, seed=_BNG_SEED)
        assert (x.nsims == 2)
        assert np.allclose(x.dataframe.loc[0, 0.0], [100.0, 0.0, 0.0])

    def test_nfsim_2initials(self):
        # Test with two initials
        A = self.model.monomers['A']
        x2 = self.sim.run(method='nf', tspan=np.linspace(0, 1),
                     initials={A(): [100, 200]}, seed=_BNG_SEED)
        assert x2.nsims == 2
        assert np.allclose(x2.dataframe.loc[0, 0.0], [100.0, 0.0, 0.0])
        assert np.allclose(x2.dataframe.loc[1, 0.0], [200.0, 0.0, 0.0])

    def test_nfsim_2params(self):
        # Test with two param_values
        x3 = self.sim.run(method='nf', tspan=np.linspace(0, 1),
                          param_values=self.param_values_2sets, seed=_BNG_SEED)
        assert x3.nsims == 2
        assert np.allclose(x3.param_values, self.param_values_2sets)

    def test_nfsim_2initials_2params(self):
        # Test with two initials and two param_values
        A = self.model.monomers['A']
        x = self.sim.run(method='nf',
                         tspan=np.linspace(0, 1),
                         initials={A(): [101, 201]},
                         param_values=[[1, 2, 3, 4, 5, 6],
                                       [7, 8, 9, 10, 11, 12]],
                         seed=_BNG_SEED)
        assert x.nsims == 2
        # Initials for A should be set by initials dict, and from param_values
        # for B and C
        assert np.allclose(x.dataframe.loc[0, 0.0], [101.0, 5.0, 6.0])
        assert np.allclose(x.dataframe.loc[1, 0.0], [201.0, 11.0, 12.0])

    @raises(ValueError)
    def test_nfsim_different_initials_lengths(self):
        A = self.model.monomers['A']
        B = self.model.monomers['B']

        sim = BngSimulator(self.model, tspan=np.linspace(0, 1),
                           initials={B(): [150, 250, 350]})
        sim.run(initials={A(): [275, 375]})

    @raises(ValueError)
    def test_nfsim_different_initials_params_lengths(self):
        A = self.model.monomers['A']

        sim = BngSimulator(self.model, tspan=np.linspace(0, 1),
                           initials={A(): [150, 250, 350]},
                           param_values=self.param_values_2sets)


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


def test_stop_if():
    Model()
    Monomer('A')
    Rule('A_synth', None >> A(), Parameter('k', 1))
    Observable('Atot', A())
    Expression('exp_const', k + 1)
    Expression('exp_dyn', Atot + 1)
    sim = BngSimulator(model, verbose=5)
    tspan = np.linspace(0, 100, 101)
    x = sim.run(tspan, stop_if='Atot>9', seed=_BNG_SEED)
    # All except the last Atot value should be <=9
    assert all(x.observables['Atot'][:-1] <= 9)
    assert x.observables['Atot'][-1] > 9
    # Starting with Atot > 9 should terminate simulation immediately
    y = sim.run(tspan, initials=x.species[-1], stop_if='Atot>9')
    assert len(y.observables) == 1


def test_set_initials_by_params():
    # This tests setting initials by changing their underlying parameter values
    # BNG Simulator uses a dictionary for initials, unlike e.g.
    # ScipyOdeSimulator, so a separate test is needed

    model = robertson.model
    t = np.linspace(0, 40, 51)
    ic_params = model.parameters_initial_conditions()
    param_values = np.array([p.value for p in model.parameters])
    ic_mask = np.array([p in ic_params for p in model.parameters])

    bng_sim = BngSimulator(model, tspan=t, verbose=0)

    # set all initial conditions to 1
    param_values[ic_mask] = np.array([1, 1, 1])
    traj = bng_sim.run(param_values=param_values)

    # set properly here
    assert np.allclose(traj.initials, [1, 1, 1])

    # overwritten in bng file. lines 196-202.
    # Values from initials_dict are used, but it should take them from
    # self.initials, so I don't see how they are getting overwritten?
    print(traj.dataframe.loc[0])
    assert np.allclose(traj.dataframe.loc[0][0:3], [1, 1, 1])

    # Same here
    param_values[ic_mask] = np.array([0, 1, 1])
    traj = bng_sim.run(param_values=param_values)
    assert np.allclose(traj.initials, [0, 1, 1])
    assert np.allclose(traj.dataframe.loc[0][0:3], [0, 1, 1])
