from pysb.testing import *
import sys
import copy
import numpy as np
from pysb import Monomer, Parameter, Initial, Observable, Rule, Expression
from pysb.simulator import ScipyOdeSimulator
from pysb.examples import robertson, earm_1_0
import unittest


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

    @raises(ValueError)
    def test_invalid_init_kwarg(self):
        ScipyOdeSimulator(self.model, tspan=self.time, spam='eggs')

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
        assert np.allclose(self.sim.initials[0][0:2],
                           [ic[1].value for ic in
                            self.model.initial_conditions])

        initials = [10, 20, 0]
        simres = self.sim.run(initials=initials)
        assert np.allclose(simres.initials[0], initials)
        assert np.allclose(simres.observables['A_free'][0], 10)

    def test_y0_as_ndarray(self):
        """Test y0 with numpy ndarray of initial conditions"""
        simres = self.sim.run(initials=np.asarray([10, 20, 0]))
        assert np.allclose(simres.observables['A_free'][0], 10)

    def test_y0_as_dictionary_monomer_species(self):
        """Test y0 with model-defined species."""
        self.sim.initials = {self.mon('A')(a=None): 17}
        base_initials = self.sim.initials
        assert base_initials[0][0] == 17

        simres = self.sim.run(initials={self.mon('A')(a=None): 10,
                               self.mon('B')(b=1) % self.mon('A')(a=1): 0,
                               self.mon('B')(b=None): 0})
        assert np.allclose(simres.initials, [10, 0, 0])
        assert np.allclose(simres.observables['A_free'][0], 10)

        # Initials should reset to base values
        assert np.allclose(self.sim.initials, base_initials)

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
        orig_param_values = self.sim.param_values
        param_values = [50, 60, 70, 0, 0]
        self.sim.param_values = param_values
        simres = self.sim.run()
        assert np.allclose(self.sim.param_values, param_values)
        assert np.allclose(simres.param_values, param_values)
        # Reset to original param values
        self.sim.param_values = orig_param_values

        # Same thing, but with a numpy array, applied as a run argument
        param_values = np.asarray([55, 65, 75, 0, 0])
        simres = self.sim.run(param_values=param_values)
        assert np.allclose(simres.param_values, param_values)
        # param_values should reset to originals after the run
        assert np.allclose(self.sim.param_values, orig_param_values)

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


class TestScipyOdeCompilerTests(TestScipySimulatorBase):
    """Test vode and analytic jacobian with different compiler backends"""
    def setUp(self):
        super(TestScipyOdeCompilerTests, self).setUp()
        self.args = {'model': self.model,
                     'tspan': self.time,
                     'integrator': 'vode',
                     'use_analytic_jacobian': True}

        self.python_sim = ScipyOdeSimulator(compiler='python', **self.args)
        self.python_res = self.python_sim.run()

    def test_cython(self):
        sim = ScipyOdeSimulator(compiler='cython', **self.args)
        simres = sim.run()
        assert simres.species.shape[0] == self.args['tspan'].shape[0]
        assert np.allclose(self.python_res.dataframe, simres.dataframe)

    def test_theano(self):
        sim = ScipyOdeSimulator(compiler='theano', **self.args)
        simres = sim.run()
        assert simres.species.shape[0] == self.args['tspan'].shape[0]
        assert np.allclose(self.python_res.dataframe, simres.dataframe)

    @unittest.skipIf(sys.version_info.major >= 3, 'weave not available for '
                                                  'Python 3')
    def test_weave(self):
        sim = ScipyOdeSimulator(compiler='weave', **self.args)
        simres = sim.run()
        assert simres.species.shape[0] == self.args['tspan'].shape[0]
        assert np.allclose(self.python_res.dataframe, simres.dataframe)


class TestScipySimulatorSequential(TestScipySimulatorBase):
    def test_sequential_initials(self):
        simres = self.sim.run()
        orig_initials = self.sim.initials

        new_initials = [10, 20, 30]
        simres = self.sim.run(initials=new_initials)

        # Check that single-run initials applied properly to the result
        assert np.allclose(simres.species[0], new_initials)
        assert np.allclose(simres.initials, new_initials)
        # Check that the single-run initials were removed after the run
        assert np.allclose(self.sim.initials, orig_initials)

    def test_sequential_initials_dict_then_list(self):
        A, B = self.model.monomers

        base_sim = ScipyOdeSimulator(
            self.model,
            initials={A(a=None): 10, B(b=None): 20})

        assert np.allclose(base_sim.initials, [10, 20, 0])
        assert len(base_sim.initials_dict) == 2

        # Now set initials using a list, which should overwrite the dict
        base_sim.initials = [30, 40, 50]

        assert np.allclose(base_sim.initials, [30, 40, 50])
        assert np.allclose(
            sorted([x[0] for x in base_sim.initials_dict.values()]),
            base_sim.initials)

    def test_sequential_param_values(self):
        orig_param_values = self.sim.param_values
        new_param_values = {'kbindAB': 0}
        new_initials = [15, 25, 35]
        simres = self.sim.run(param_values=new_param_values,
                              initials=new_initials)
        # No new AB_complex should be formed
        assert np.allclose(simres.observables['AB_complex'], new_initials[2])
        assert simres.nsims == 1
        # Original param_values should be restored after run
        assert np.allclose(self.sim.param_values, orig_param_values)

        # Check that per-run param override works when a base param
        # dictionary is also specified
        self.sim.param_values = new_param_values
        base_param_values = new_param_values
        new_param_values = {'ksynthB': 50}
        simres = self.sim.run(param_values=new_param_values)
        # Check that new param value override applied
        assert np.allclose(simres.param_values[0][1],
                           new_param_values['ksynthB'])
        # Check that simulator reverts to base param values
        assert np.allclose(self.sim.param_values[0][2],
                           base_param_values['kbindAB'])
        # Reset to original param values
        self.sim.param_values = orig_param_values

    def test_sequential_tspan(self):
        tspan = np.linspace(0, 10, 11)
        orig_tspan = self.sim.tspan
        simres = self.sim.run(tspan=tspan)
        # Check that new tspan applied properly
        assert np.allclose(simres.tout, tspan)
        # Check that simulator instance reset to original tspan
        assert np.allclose(self.sim.tspan, orig_tspan)


class TestScipySimulatorMultiple(TestScipySimulatorBase):
    def test_initials_and_param_values_two_lists(self):
        initials = [[10, 20, 30], [50, 60, 70]]
        param_values = [[55, 65, 75, 0, 0],
                        [90, 100, 110, 5, 6]]
        import pysb.bng
        pysb.bng.generate_equations(self.sim.model)
        simres = self.sim.run(initials=initials, param_values=param_values)
        assert np.allclose(simres.species[0][0], initials[0])
        assert np.allclose(simres.species[1][0], initials[1])

        assert np.allclose(simres.param_values[0], param_values[0])
        assert np.allclose(simres.param_values[1], param_values[1])

        assert simres.nsims == 2

        # Check the methods underlying these properties work
        df = simres.dataframe
        all = simres.all

        # Try overriding above lists of initials/params with dicts
        self.sim.initials = initials
        self.sim.param_values = param_values
        simres = self.sim.run(
            initials={self.mon('A')(a=None): [103, 104]},
            param_values={'ksynthA': [101, 102]})
        # Simulator initials and params should not persist run() overrides
        assert np.allclose(self.sim.initials, initials)
        assert np.allclose(self.sim.param_values, param_values)
        # Create the expected initials/params arrays and compare to result
        initials = np.array(initials)
        initials[:, 0] = [103, 104]
        param_values = np.array(param_values)
        param_values[:, 0] = [101, 102]
        assert np.allclose(simres.initials, initials)
        assert np.allclose(simres.param_values, param_values)

    @raises(ValueError)
    def test_run_initials_different_length_to_base(self):
        initials = [[10, 20, 30, 40], [50, 60, 70, 80]]
        self.sim.initials = initials
        self.sim.run(initials=initials[0])

    @raises(ValueError)
    def test_run_params_different_length_to_base(self):
        param_values = [[55, 65, 75, 0, 0, 1],
                        [90, 100, 110, 5, 6, 7]]
        self.sim.param_values = param_values
        self.sim.run(param_values=param_values[0])

    def test_param_values_dict(self):
        param_values = {'A_init': [0, 100]}
        initials = {self.model.monomers['B'](b=None): [250, 350]}

        simres = self.sim.run(param_values=param_values)
        assert np.allclose(simres.dataframe.loc[(slice(None), 0.0), 'A_free'],
                           [0, 100])

        simres = self.sim.run(param_values={'B_init': [200, 300]})
        assert np.allclose(simres.dataframe.loc[(slice(None), 0.0), 'A_free'],
                           [0, 0])
        assert np.allclose(simres.dataframe.loc[(slice(None), 0.0), 'B_free'],
                           [200, 300])

        simres = self.sim.run(initials=initials, param_values=param_values)
        assert np.allclose(simres.dataframe.loc[(slice(None), 0.0), 'A_free'],
                           [0, 100])
        assert np.allclose(simres.dataframe.loc[(slice(None), 0.0), 'B_free'],
                           [250, 350])

    @raises(ValueError)
    def test_initials_and_param_values_differing_lengths(self):
        initials = [[10, 20, 30, 40], [50, 60, 70, 80]]
        param_values = [[55, 65, 75, 0, 0],
                        [90, 100, 110, 5, 6],
                        [90, 100, 110, 5, 6]]
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


def test_set_initial_to_zero():
    sim = ScipyOdeSimulator(robertson.model, tspan=np.linspace(0, 100))
    simres = sim.run(initials={robertson.model.monomers['A'](): 0})
    assert np.allclose(simres.observables['A_total'], 0)


def test_robertson_integration():
    """Ensure robertson model simulates."""
    t = np.linspace(0, 100)
    # Run with or without inline
    sim = ScipyOdeSimulator(robertson.model)
    simres = sim.run(tspan=t)
    assert simres.species.shape[0] == t.shape[0]
    if sim._compiler != 'python':
        # Also run without inline
        sim = ScipyOdeSimulator(robertson.model, tspan=t, compiler='python')
        simres = sim.run()
        assert simres.species.shape[0] == t.shape[0]


def test_earm_integration():
    """Ensure earm_1_0 model simulates."""
    t = np.linspace(0, 1e3)
    # Run with or without inline
    sim = ScipyOdeSimulator(earm_1_0.model, tspan=t)
    sim.run()
    if sim._compiler != 'python':
        # Also run without inline
        ScipyOdeSimulator(earm_1_0.model, tspan=t, compiler='python').run()


@raises(ValueError)
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
    simres.all
    simres.dataframe


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
    simres.all
    simres.dataframe


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

