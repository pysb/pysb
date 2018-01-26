from pysb.testing import *
from pysb import *
from pysb.bng import *
import os
import unittest
from pysb.core import as_complex_pattern


@with_model
def test_generate_network():
    Monomer('A')
    assert_raises((NoInitialConditionsError, NoRulesError),
                  generate_network, model)
    Parameter('A_0', 1)
    Initial(A(), A_0)
    assert_raises(NoRulesError, generate_network, model)
    Parameter('k', 1)
    Rule('degrade', A() >> None, k)
    ok_(generate_network(model))


@unittest.skipIf(os.name == 'nt', 'BNG Console does not work on Windows')
@with_model
def test_simulate_network_console():
    Monomer('A')
    Parameter('A_0', 1)
    Initial(A(), A_0)
    Parameter('k', 1)
    Rule('degrade', A() >> None, k)
    # Suppress network overwrite warning from simulate command
    with BngConsole(model, suppress_warnings=True) as bng:
        bng.generate_network(overwrite=True)
        bng.action('simulate', method='ssa', t_end=20000, n_steps=100)


@with_model
def test_sequential_simulations():
    Monomer('A')
    Parameter('A_0', 1)
    Initial(A(), A_0)
    Parameter('k', 1)
    Rule('degrade', A() >> None, k)
    # Suppress network overwrite warning from simulate command
    with BngFileInterface(model) as bng:
        bng.action('generate_network')
        bng.action('simulate', method='ssa', t_end=20000, n_steps=100)
        bng.execute()
        yfull1 = bng.read_simulation_results()
        ok_(yfull1.size == 101)

        # Run another simulation by reloading the existing network file
        bng.action('simulate', method='ssa', t_end=10000, n_steps=50)
        bng.execute(reload_netfile=True)
        yfull2 = bng.read_simulation_results()
        ok_(yfull2.size == 51)


@with_model
def test_compartment_species_equivalence():
    Parameter('p', 1)
    Monomer('Q', ['x'])
    Monomer('R', ['y'])
    Compartment('C', None, 3, p)
    Rule('bind', Q(x=None) + R(y=None) >> Q(x=1) % R(y=1), p)
    Initial(Q(x=None) ** C, p)
    Initial(R(y=None) ** C, p)
    generate_equations(model)
    for i, (cp, param) in enumerate(model.initial_conditions):
        ok_(cp.is_equivalent_to(model.species[i]))
    ok_(model.species[2].is_equivalent_to(Q(x=1) ** C % R(y=1) ** C))


@with_model
def test_bidirectional_rules_collapse():
    Monomer('A')
    Monomer('B')
    Initial(B(), Parameter('B_init', 0))
    Initial(A(), Parameter('A_init', 100))
    Rule('Rule1', B() | A(), Parameter('k3', 10), Parameter('k1', 1))
    Rule('Rule2', A() | B(), k1, Parameter('k2', 1))
    Rule('Rule3', B() >> A(), Parameter('k4', 5))
    generate_equations(model)
    ok_(len(model.reactions) == 4)
    ok_(len(model.reactions_bidirectional) == 1)
    ok_(len(model.reactions_bidirectional[0]['rule']) == 3)
    ok_(model.reactions_bidirectional[0]['reversible'])


@with_model
def test_bidirectional_rules():
    Monomer('A')
    Monomer('B')
    Initial(A(), Parameter('A_init', 100))
    Rule('Rule1', A() | B(), Parameter('k1', 1), Parameter('k2', 1))
    Rule('Rule2', B() >> A(), Parameter('k3', 10))
    Rule('Rule3', B() >> A(), Parameter('k4', 5))
    generate_equations(model)
    ok_(len(model.reactions) == 4)
    ok_(len(model.reactions_bidirectional) == 1)
    ok_(len(model.reactions_bidirectional[0]['rule']) == 3)
    ok_(model.reactions_bidirectional[0]['reversible'])
    #TODO Check that 'rate' has 4 terms


@with_model
def test_zero_order_synth_no_initials():
    Monomer('A')
    Monomer('B')
    Rule('Rule1', None >> A(), Parameter('ksynth', 100))
    Rule('Rule2', A() | B(), Parameter('kf', 10), Parameter('kr', 1))
    generate_equations(model)


@with_model
def test_reversible_synth_deg():
    Monomer('A')
    Parameter('k_synth', 2.0)
    Parameter('k_deg', 1.0)
    Rule('synth_deg', A() | None, k_deg, k_synth)
    assert synth_deg.is_synth()
    assert synth_deg.is_deg()
    generate_equations(model)


@with_model
def test_nfsim():
    Monomer('A', ['a'])
    Monomer('B', ['b'])

    Parameter('ksynthA', 100)
    Parameter('ksynthB', 100)
    Parameter('kbindAB', 100)

    Parameter('A_init', 20)
    Parameter('B_init', 30)

    Initial(A(a=None), A_init)
    Initial(B(b=None), B_init)

    Observable("A_free", A(a=None))
    Observable("B_free", B(b=None))
    Observable("AB_complex", A(a=1) % B(b=1))

    Rule('A_synth', None >> A(a=None), ksynthA)
    Rule('B_synth', None >> B(b=None), ksynthB)
    Rule('AB_bind', A(a=None) + B(b=None) >> A(a=1) % B(b=1), kbindAB)

    with BngFileInterface(model) as bng:
        bng.action('simulate', method='nf', t_end=1000, n_steps=100)
        bng.execute()
        res = bng.read_simulation_results()
        assert res.dtype.names == ('time', 'A_free', 'B_free', 'AB_complex')
        assert len(res) == 101


@with_model
def test_unicode_strs():
    Monomer(u'A', [u'b'], {u'b':[u'y', u'n']})
    Monomer(u'B')
    Rule(u'rule1', A(b=u'y') >> B(), Parameter(u'k', 1))
    Initial(A(b=u'y'), Parameter(u'A_0', 100))
    generate_equations(model)


@with_model
def test_none_in_rxn_pat():
    Monomer(u'A', [u'b'], {u'b': [u'y', u'n']})
    Monomer(u'B')
    Rule(u'rule1', A(b=u'y') + None >> None + B(),
         Parameter(u'k', 1))
    Initial(A(b=u'y'), Parameter(u'A_0', 100))
    generate_equations(model)


@with_model
def test_sympy_parameter_keyword():
    Monomer('A')
    Initial(A(), Parameter('A_0', 100))
    Parameter('deg', 10)  # deg is a sympy function
    Rule('Rule1', A() >> None, deg)
    generate_equations(model)
