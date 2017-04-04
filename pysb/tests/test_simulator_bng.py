from pysb.testing import *
import numpy as np
from pysb import Monomer, Parameter, Initial, Observable, Rule
from pysb.simulator.bng_ssa import BngSimulator
from pysb.bng import generate_equations


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
        assert x.all.shape == (51,)

    def test_multi_simulations(self):
        x = self.sim.run(n_sim=10)
        assert np.shape(x.observables) == (10, 51)

    def test_change_parameters(self):
        x = self.sim.run(n_sim=10, param_values={'ksynthA': 200},
                         initials={self.model.species[0]: 100})
        species = np.array(x.all)
        assert species[0][0][0] == 100.

    def test_nfsim(self):
        x = self.sim.run(n_sim=2, method='nf')
        observables = np.array(x.observables)
        assert np.shape(observables) == (2, 51)

    def tearDown(self):
        self.model = None
        self.time = None
        self.sim = None
