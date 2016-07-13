from pysb.testing import *
from pysb.integrate import odesolve, Solver
import numpy as np
from pysb import Monomer, Parameter, Initial, Observable, Rule, Expression
from pysb.bng import run_ssa

from pysb.examples import robertson, earm_1_0


class TestSolver(object):

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
        self.solver = Solver(self.model, self.time, integrator='vode')

    def tearDown(self):
        self.model = None
        self.time = None
        self.solver = None

    def test_vode_solver_run(self):
        """Test vode."""
        self.solver.run()

    def test_vode_jac_solver_run(self):
        """Test vode and analytic jacobian."""
        solver_vode_jac = Solver(self.model, self.time, integrator='vode',
                                  use_analytic_jacobian=True)
        solver_vode_jac.run()

    def test_lsoda_solver_run(self):
        """Test lsoda."""
        solver_lsoda = Solver(self.model, self.time, integrator='lsoda')
        solver_lsoda.run()

    def test_lsoda_jac_solver_run(self):
        """Test lsoda and analytic jacobian."""
        solver_lsoda_jac = Solver(self.model, self.time, integrator='lsoda',
                                  use_analytic_jacobian=True)
        solver_lsoda_jac.run()

    @raises(NotImplementedError)
    def test_y0_as_dictionary_monomer_species(self):
        """Test y0 with model-defined species."""
        self.solver.run(y0={"A(a=None)": 10, "B(b=1) % A(a=1)": 0,
                        "B(b=None)": 0})
        assert np.allclose(self.solver.y[0, 0], 10)

    @raises(NotImplementedError)
    def test_y0_as_dictionary_with_bound_species(self):
        """Test y0 with dynamically generated species."""
        self.solver.run(y0={"A(a=None)": 0, "B(b=1) % A(a=1)": 100,
                        "B(b=None)": 0})
        assert np.allclose(self.solver.y[0, 3], 100)

    @raises(NotImplementedError)
    def test_y0_invalid_dictionary_key(self):
        """Test y0 with invalid monomer name."""
        self.solver.run(y0={"C(c=None)": 1})

    @raises(NotImplementedError)
    def test_y0_non_numeric_value(self):
        """Test y0 with non-numeric value."""
        self.solver.run(y0={"A(a=None)": 'eggs'})

    def test_param_values_as_dictionary(self):
        """Test param_values as a dictionary."""
        self.solver.run(param_values={'kbindAB': 0})
        # kbindAB=0 should ensure no AB_complex is produced.
        assert np.allclose(self.solver.yobs["AB_complex"], 0)

    @raises(IndexError)
    def test_param_values_invalid_dictionary_key(self):
        """Test param_values with invalid parameter name."""
        self.solver.run(param_values={'spam': 150})

    @raises(ValueError, TypeError)
    def test_param_values_non_numeric_value(self):
        """Test param_values with non-numeric value."""
        self.solver.run(param_values={'ksynthA': 'eggs'})


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
    x = odesolve(model, time)


def test_robertson_integration():
    """Ensure robertson model simulates."""
    t = np.linspace(0, 100)
    # Run with or without inline
    sol = Solver(robertson.model, t)
    sol.run()
    assert sol.y.shape[0] == t.shape[0]
    if Solver._use_inline:
        # Also run without inline
        Solver._use_inline = False
        sol = Solver(robertson.model, t)
        sol.run()
        assert sol.y.shape[0] == t.shape[0]
        Solver._use_inline = True


def test_earm_integration():
    """Ensure earm_1_0 model simulates."""
    t = np.linspace(0, 1e3)
    # Run with or without inline
    sol = Solver(earm_1_0.model, t)
    sol.run()
    if Solver._use_inline:
        # Also run without inline
        Solver._use_inline = False
        sol = Solver(earm_1_0.model, t)
        sol.run()
        Solver._use_inline = True


def test_run_ssa():
    """Test run_ssa."""
    run_ssa(robertson.model, t_end=20000, n_steps=100, verbose=False)


@raises(UserWarning)
def test_nonexistent_integrator():
    """Ensure nonexistent integrator raises."""
    Solver(robertson.model, np.linspace(0, 1, 2), integrator='does_not_exist')

def test_initialize_tspan():
    """Test setting the time span for simulation in constructor."""
    ts = np.linspace(0, 10, 10)
    sol = Solver(robertson.model, ts)
    assert len(ts) == sol.y.shape[0]
    assert len(ts) == sol.yobs.shape[0]
    assert len(ts) == sol.yexpr.shape[0]

def test_set_tspan():
    """Test setting the time span for simulation."""
    ts1 = np.linspace(0, 10, 10)
    ts2 = np.linspace(0, 10, 100)
    sol = Solver(robertson.model, ts1)
    sol.set_tspan(ts2)
    assert len(ts2) == sol.y.shape[0]
    assert len(ts2) == sol.yobs.shape[0]
    assert len(ts2) == sol.yexpr.shape[0]
    # Smoke test to make sure solver runs
    sol.run()
