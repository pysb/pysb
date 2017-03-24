from pysb.testing import *
import sys
import copy
import numpy as np
from pysb import Monomer, Parameter, Initial, Observable, Rule, Expression
from pysb.simulator import ScipyOdeSimulator, SimulatorException
from pysb.examples import robertson, earm_1_0


class TestScipySimulatorBase(object):
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

        # Convenience shortcut for accessing model monomer objects
        self.mon = lambda m: self.model.monomers[m]

        # This timespan is chosen to be enough to trigger a Jacobian evaluation
        # on the various solvers.
        self.time = np.linspace(0, 1)
        self.sim = ScipyOdeSimulator(self.model, tspan=self.time,
                                     integrator='vode')

    def tearDown(self):
        self.model = None
        self.time = None
        self.sim = None


class TestScipySimulatorSingle(TestScipySimulatorBase):
    def test_vode_solver_run(self):
        """Test vode."""
        simres = self.sim.run()
        assert simres._nsims == 1

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

    def test_y0_as_list(self):
        """Test y0 with list of initial conditions"""
        # Test the initials getter method before anything is changed
        assert np.allclose(self.sim.initials[0][0:3],
                           [ic[1].value for ic in
                            self.model.initial_conditions])

        initials = [10, 20, 0, 0]
        simres = self.sim.run(initials=initials)
        assert np.allclose(self.sim.initials[0], initials)
        assert np.allclose(simres.observables['A_free'][0], 10)

    def test_y0_as_ndarray(self):
        """Test y0 with numpy ndarray of initial conditions"""
        simres = self.sim.run(initials=np.asarray([10, 20, 0, 0]))
        assert np.allclose(simres.observables['A_free'][0], 10)

    def test_y0_as_dictionary_monomer_species(self):
        """Test y0 with model-defined species."""
        simres = self.sim.run(initials={self.mon('A')(a=None): 10,
                               self.mon('B')(b=1) % self.mon('A')(a=1): 0,
                               self.mon('B')(b=None): 0})
        assert np.allclose(self.sim.initials, [10, 0, 1, 0])
        assert np.allclose(simres.observables['A_free'][0], 10)

    def test_y0_as_dictionary_with_bound_species(self):
        """Test y0 with dynamically generated species."""
        simres = self.sim.run(initials={self.mon('A')(a=None): 0,
                               self.mon('B')(b=1) % self.mon('A')(a=1): 100,
                               self.mon('B')(b=None): 0})
        assert np.allclose(simres.observables['AB_complex'][0], 100)

    @raises(TypeError)
    def test_y0_non_numeric_value(self):
        """Test y0 with non-numeric value."""
        self.sim.run(initials={self.mon('A')(a=None): 'eggs'})

    def test_param_values_as_dictionary(self):
        """Test param_values as a dictionary."""
        simres = self.sim.run(param_values={'kbindAB': 0})
        # kbindAB=0 should ensure no AB_complex is produced.
        assert np.allclose(simres.observables["AB_complex"], 0)

    def test_param_values_as_list_ndarray(self):
        """Test param_values as a list and ndarray."""
        param_values = [50, 60, 70, 0, 0, 1]
        self.sim.run(param_values=param_values)
        assert np.allclose(self.sim.param_values, param_values)
        # Same thing, but with a numpy array
        param_values = np.asarray([55, 65, 75, 0, 0, 1])
        self.sim.run(param_values=param_values)
        assert np.allclose(self.sim.param_values, param_values)

    @raises(IndexError)
    def test_param_values_invalid_dictionary_key(self):
        """Test param_values with invalid parameter name."""
        self.sim.run(param_values={'spam': 150})

    @raises(ValueError, TypeError)
    def test_param_values_non_numeric_value(self):
        """Test param_values with non-numeric value."""
        self.sim.run(param_values={'ksynthA': 'eggs'})

    def test_result_dataframe(self):
        df = self.sim.run().dataframe


class TestScipySimulatorSequential(TestScipySimulatorBase):
    def test_sequential_runs(self):
        simres = self.sim.run()

        new_initials = [10, 20, 30, 40]
        simres = self.sim.run(initials=new_initials)
        assert np.allclose(simres.species[0], new_initials)

        new_param_values = {'kbindAB': 0}
        simres = self.sim.run(param_values=new_param_values)
        # No new AB_complex should be formed
        assert np.allclose(simres.observables['AB_complex'], 40)
        assert simres.nsims == 1


class TestScipySimulatorMultiple(TestScipySimulatorBase):
    def test_initials_and_param_values_two_lists(self):
        initials = [[10, 20, 30, 40], [50, 60, 70, 80]]
        param_values = [[55, 65, 75, 0, 0, 1],
                        [90, 100, 110, 5, 6, 7]]
        simres = self.sim.run(initials=initials, param_values=param_values)
        assert np.allclose(simres.species[0][0], initials[0])
        assert np.allclose(simres.species[1][0], initials[1])
        assert np.allclose(self.sim.param_values[0], param_values[0])
        assert np.allclose(self.sim.param_values[1], param_values[1])

        assert simres.nsims == 2

        # Check the methods underlying these properties work
        df = simres.dataframe
        all = simres.all

    @raises(SimulatorException)
    def test_initials_and_param_values_differing_lengths(self):
        initials = [[10, 20, 30, 40], [50, 60, 70, 80]]
        param_values = [[55, 65, 75, 0, 0, 1],
                        [90, 100, 110, 5, 6, 7],
                        [90, 100, 110, 5, 6, 7]]
        self.sim.run(initials=initials, param_values=param_values)


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
    sim = ScipyOdeSimulator(model, tspan=time)
    simres = sim.run()
    keff_vals = simres.expressions['keff']
    assert len(keff_vals) == len(time)
    assert np.allclose(keff_vals, 1.8181818181818182e-05)


def test_robertson_integration():
    """Ensure robertson model simulates."""
    t = np.linspace(0, 100)
    # Run with or without inline
    sim = ScipyOdeSimulator(robertson.model)
    simres = sim.run(tspan=t)
    assert simres.species.shape[0] == t.shape[0]
    if ScipyOdeSimulator._use_inline:
        # Also run without inline
        ScipyOdeSimulator._use_inline = False
        sim = ScipyOdeSimulator(robertson.model, tspan=t)
        simres = sim.run()
        assert simres.species.shape[0] == t.shape[0]
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
        ScipyOdeSimulator(earm_1_0.model, tspan=t).run()
        ScipyOdeSimulator._use_inline = True


@raises(SimulatorException)
def test_simulation_no_tspan():
    ScipyOdeSimulator(robertson.model).run()


@raises(UserWarning)
def test_nonexistent_integrator():
    """Ensure nonexistent integrator raises."""
    ScipyOdeSimulator(robertson.model, tspan=np.linspace(0, 1, 2),
                      integrator='does_not_exist')


def test_unicode_obsname_ascii():
    """Ensure ascii-convetible unicode observable names are handled."""
    t = np.linspace(0, 100)
    rob_copy = copy.deepcopy(robertson.model)
    rob_copy.observables[0].name = u'A_total'
    sim = ScipyOdeSimulator(rob_copy)
    simres = sim.run(tspan=t)


if sys.version_info[0] < 3:
    @raises(ValueError)
    def test_unicode_obsname_nonascii():
        """Ensure non-ascii unicode observable names error in python 2."""
        t = np.linspace(0, 100)
        rob_copy = copy.deepcopy(robertson.model)
        rob_copy.observables[0].name = u'A_total\u1234'
        sim = ScipyOdeSimulator(rob_copy)
        simres = sim.run(tspan=t)


def test_unicode_exprname_ascii():
    """Ensure ascii-convetible unicode expression names are handled."""
    t = np.linspace(0, 100)
    rob_copy = copy.deepcopy(robertson.model)
    ab = rob_copy.observables['A_total'] + rob_copy.observables['B_total']
    expr = Expression(u'A_plus_B', ab, _export=False)
    rob_copy.add_component(expr)
    sim = ScipyOdeSimulator(rob_copy)
    simres = sim.run(tspan=t)


if sys.version_info[0] < 3:
    @raises(ValueError)
    def test_unicode_exprname_nonascii():
        """Ensure non-ascii unicode expression names error in python 2."""
        t = np.linspace(0, 100)
        rob_copy = copy.deepcopy(robertson.model)
        ab = rob_copy.observables['A_total'] + rob_copy.observables['B_total']
        expr = Expression(u'A_plus_B\u1234', ab, _export=False)
        rob_copy.add_component(expr)
        sim = ScipyOdeSimulator(rob_copy)
        simres = sim.run(tspan=t)

