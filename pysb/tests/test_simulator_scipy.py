from pysb.testing import *
from pysb.integrate import Solver
import numpy as np
from pysb import Monomer, Parameter, Initial, Observable, Rule, Expression
from pysb.simulator import ScipyOdeSimulator
from pysb.examples import robertson, earm_1_0


class TestScipySimulator(object):
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

        # This timespan is chosen to be enough to trigger a Jacobian evaluation
        # on the various solvers.
        self.time = np.linspace(0, 1)
        self.sim = ScipyOdeSimulator(self.model, tspan=self.time,
                                     integrator='vode')

    def tearDown(self):
        self.model = None
        self.time = None
        self.sim = None

    def test_vode_solver_run(self):
        """Test vode."""
        self.sim.run()

    def test_vode_jac_solver_run(self):
        """Test vode and analytic jacobian."""
        solver_vode_jac = ScipyOdeSimulator(self.model, tspan=self.time,
                                            integrator='vode',
                                            use_analytic_jacobian=True)
        solver_vode_jac.run()

    def test_lsoda_solver_run(self):
        """Test lsoda."""
        solver_lsoda = ScipyOdeSimulator(self.model, tspan=self.time,
                                         integrator='lsoda')
        solver_lsoda.run()

    def test_lsoda_jac_solver_run(self):
        """Test lsoda and analytic jacobian."""
        solver_lsoda_jac = ScipyOdeSimulator(self.model, tspan=self.time,
                                             integrator='lsoda',
                                             use_analytic_jacobian=True)
        solver_lsoda_jac.run()

    @raises(NotImplementedError)
    def test_y0_as_dictionary_monomer_species(self):
        """Test y0 with model-defined species."""
        self.sim.run(initials={"A(a=None)": 10, "B(b=1) % A(a=1)": 0,
                               "B(b=None)": 0})
        assert np.allclose(self.sim.concs_all()[0, 0], 10)

    @raises(NotImplementedError)
    def test_y0_as_dictionary_with_bound_species(self):
        """Test y0 with dynamically generated species."""
        self.sim.run(initials={"A(a=None)": 0, "B(b=1) % A(a=1)": 100,
                                 "B(b=None)": 0})
        assert np.allclose(self.sim.concs_all()[0, 3], 100)

    @raises(NotImplementedError)
    def test_y0_invalid_dictionary_key(self):
        """Test y0 with invalid monomer name."""
        self.sim.run(initials={"C(c=None)": 1})

    @raises(NotImplementedError)
    def test_y0_non_numeric_value(self):
        """Test y0 with non-numeric value."""
        self.sim.run(initials={"A(a=None)": 'eggs'})

    def test_param_values_as_dictionary(self):
        """Test param_values as a dictionary."""
        self.sim.run(param_values={'kbindAB': 0})
        # kbindAB=0 should ensure no AB_complex is produced.
        assert np.allclose(self.sim.concs_observables()["AB_complex"], 0)

    @raises(IndexError)
    def test_param_values_invalid_dictionary_key(self):
        """Test param_values with invalid parameter name."""
        self.sim.run(param_values={'spam': 150})

    @raises(ValueError, TypeError)
    def test_param_values_non_numeric_value(self):
        """Test param_values with non-numeric value."""
        self.sim.run(param_values={'ksynthA': 'eggs'})


@with_model
def test_integrate_with_expression():
    """Ensure a model with Expressions simulates."""

    Monomer('s1')
    Monomer('s9')
    Monomer('s16')
    Monomer('s20')

    # Parameters should be able to contain s(\d+) without error
    Parameter('ks0',2e-5)
    Parameter('ka20', 1e5)

    Initial(s9(), Parameter('s9_0', 10000))

    Observable('s1_obs', s1())
    Observable('s9_obs', s9())
    Observable('s16_obs', s16())
    Observable('s20_obs', s20())

    Expression('keff', (ks0*ka20)/(ka20+s9_obs))

    Rule('R1', None >> s16(), ks0)
    Rule('R2', None >> s20(), ks0)
    Rule('R3', s16() + s20() >> s16() + s1(), keff)

    time = np.linspace(0, 40)
    x = ScipyOdeSimulator.execute(model, tspan=time)


def test_robertson_integration():
    """Ensure robertson model simulates."""
    t = np.linspace(0, 100)
    # Run with or without inline
    sim = ScipyOdeSimulator(robertson.model)
    sim.run(tspan=t)
    assert sim.concs_species().shape[0] == t.shape[0]
    if ScipyOdeSimulator._use_inline:
        # Also run without inline
        ScipyOdeSimulator._use_inline = False
        sim = ScipyOdeSimulator(robertson.model, tspan=t)
        sim.run()
        assert sim.concs_species().shape[0] == t.shape[0]
        ScipyOdeSimulator._use_inline = True


def test_earm_integration():
    """Ensure earm_1_0 model simulates."""
    t = np.linspace(0, 1e3)
    # Run with or without inline
    sim = ScipyOdeSimulator(earm_1_0.model, tspan=t)
    sim.run()
    if ScipyOdeSimulator._use_inline:
        # Also run without inline
        ScipyOdeSimulator._use_inline = False
        ScipyOdeSimulator.execute(earm_1_0.model, tspan=t)
        ScipyOdeSimulator._use_inline = True


@raises(UserWarning)
def test_nonexistent_integrator():
    """Ensure nonexistent integrator raises."""
    ScipyOdeSimulator(robertson.model, tspan=np.linspace(0, 1, 2),
                      integrator='does_not_exist')
