from pysb.testing import *
from pysb.core import *
from functools import partial
import itertools


def test_component_names_valid():
    for name in 'a', 'B', 'AbC', 'dEf', '_', '_7', '__a01b__999x_x___':
        c = Component(name, _export=False)
        eq_(c.name, name)

def test_component_names_invalid():
    for name in 'a!', '!B', 'A!bC~`\\', '_!', '_!7', '__a01b  999x_x___!':
        assert_raises(InvalidComponentNameError, Component, name, _export=False)

def test_monomer():
    sites = ['x', 'y', 'z']
    states = {'y': ['foo', 'bar', 'baz'], 'x': ['e']}
    m = Monomer('A', sites, states, _export=False)
    assert_equal(m.sites, sites)
    assert_equal(m.site_states, states)
    assert_equal(type(m()), MonomerPattern)

    assert_raises(ValueError, Monomer, 'A', 'x', _export=False)
    assert_raises(Exception, Monomer, 'A', 'x', 'x', _export=False)
    assert_raises(Exception, Monomer, 'A', ['x'], {'y': ['a']}, _export=False)
    assert_raises(Exception, Monomer, 'A', ['x'], {'x': [1]}, _export=False)

@with_model
def test_monomer_model():
    Monomer('A')
    ok_(A in model.monomers)
    ok_(A in model.all_components())
    ok_(A not in model.all_components() - model.monomers)


@with_model
def test_initial():
    Monomer('A', ['s'])
    Parameter('A_0')
    Initial(A(s=None), A_0)
    assert_raises_iice = partial(assert_raises, InvalidInitialConditionError,
                                 Initial)
    assert_raises_iice('not a complexpattern', A_0)
    assert_raises_iice(A(), A_0)
    assert_raises_iice(A(s=None), A_0)
    assert_raises_iice(MatchOnce(A(s=None)), A_0)

@with_model
def test_model_pickle():
    import pickle
    A = Monomer('A', _export=False)
    B = Monomer('B', ['x', 'y'], {'x': ['e', 'f']}, _export=False)
    k = Parameter('k', 1.0, _export=False)
    r = Rule('r', A() + B(x='e', y=WILD) >> A() % B(x='f', y=None), k,
             _export=False)
    o = Observable('o', A() % B(), _export=False)
    e = Expression('e', k * o, _export=False)
    c = Compartment('c', None, 3, k, _export=False)
    for comp in [A, B, k, r, o, e, c]:
        model.add_component(comp)
    model.add_component(c)
    Initial(A() ** c, k)
    assert_equal(len(model.all_components()), 7)
    model2 = pickle.loads(pickle.dumps(model))
    check_model_against_component_list(model, model2.all_components())

@with_model
def test_monomer_as_reaction_pattern():
    A = Monomer('A', _export=False)
    as_reaction_pattern(A)

@with_model
def test_monomer_as_complex_pattern():
    A = Monomer('A', _export=False)
    as_complex_pattern(A)

@with_model
def test_monomerpattern():
    A = Monomer('A',sites=['a'],site_states={'a':['u','p']},_export=False)
    Aw = A(a=('u', ANY))

@with_model
def test_observable_constructor_with_monomer():
    A = Monomer('A', _export=False)
    o = Observable('o', A, _export=False)

@with_model
def test_compartment_initial_error():
    Monomer('A', ['s'])
    Parameter('A_0', 2.0)
    c1 = Compartment("C1")
    c2 = Compartment("C2")
    Initial(A(s=None)**c1, A_0)
    Initial(A(s=None)**c2, A_0)

@with_model
def test_monomer_pattern_add_to_none():
    """Ensure that MonomerPattern + None returns a ReactionPattern."""
    Monomer('A', ['s'])
    ok_(isinstance(A() + None, ReactionPattern),
        'A() + None did not return a ReactionPattern.')

@with_model
def test_complex_pattern_add_to_none():
    """Ensure that ComplexPattern + None returns a ReactionPattern."""
    Monomer('A', ['s'])
    ok_(isinstance(A(s=1) % A(s=1) + None, ReactionPattern),
        'A(s=1) % A(s=1) + None did not return a ReactionPattern.')

@with_model
def test_reaction_pattern_add_to_none():
    """Ensure that ReactionPattern + None returns a ReactionPattern."""
    Monomer('A', ['s'])
    cp = A(s=1) % A(s=1)
    rp = cp + cp
    ok_(isinstance(rp + None, ReactionPattern),
        'ReactionPattern + None did not return a ReactionPattern.')

@with_model
def test_complex_pattern_call():
    """Ensure ComplexPattern calling (refinement) works as expected."""
    Monomer('A', ['w', 'x'], {'x': ('e', 'f')})
    Monomer('B', ['y', 'z'], {'z': ('g', 'h')})
    cp = A(w=1, x='e') % B(y=1, z='g')
    r = {'x': 'f', 'z': 'h'} # refinement for complexpattern
    ok_(cp(**r))
    ok_(cp(**r).monomer_patterns[0].site_conditions['x'] == r['x'])
    ok_(cp(r))
    ok_(cp(r).monomer_patterns[0].site_conditions['x'] == r['x'])
    assert_raises(RedundantSiteConditionsError, cp, {'x': 'f'}, z='h')

@with_model
def test_monomer_unicode():
    Monomer(u'A', [u's'], {u's': [u's1', u's2']})


def _check_pattern_equivalence(complex_pattern_list, equivalent=True):
    """ Check all complex pattern permutations are equivalent (or not) """
    for cp0, cp1 in itertools.permutations(complex_pattern_list, 2):
        assert cp0.is_equivalent_to(cp1) == equivalent


@with_model
def test_complex_pattern_equivalence_compartments():
    Monomer('A')
    Monomer('B')

    Compartment('C1')
    Compartment('C2')

    cp0 = ComplexPattern([A()], compartment=C1)  # Only species compartment
    cp1 = as_complex_pattern(A() ** C1)          # Only monomer compartment
    cp2 = ComplexPattern([A() ** C1], compartment=C1)  # Both compartments

    _check_pattern_equivalence((cp0, cp1, cp2))

    cp3 = (A() % B()) ** C1
    cp4 = A() ** C1 % B() ** C1
    cp5 = (A() ** C1 % B()) ** C1
    # Species compartment is ignored if all monomer patterns have a compartment
    cp6 = (A() ** C1 % B() ** C1) ** C2

    _check_pattern_equivalence((cp3, cp4, cp5, cp6))

    # Species compartment should not override a specified monomer compartment
    cp7 = (A() ** C1 % B() ** C2) ** C1
    _check_pattern_equivalence((cp5, cp7), equivalent=False)

    # Test enable_synth_deg creates two initial conditions, one for each
    # compartment
    model.enable_synth_deg()
    assert len(model.initial_conditions) == 2

    # Check that enable_synth_deg is idempotent
    model.enable_synth_deg()
    assert len(model.initial_conditions) == 2
