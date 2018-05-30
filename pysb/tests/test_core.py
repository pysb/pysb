from pysb.testing import *
from pysb.core import *
from functools import partial
from nose.tools import assert_raises


@with_model
def test_component_names_valid():
    for name in 'a', 'B', 'AbC', 'dEf', '_', '_7', '__a01b__999x_x___':
        c = Monomer(name, _export=False)
        eq_(c.name, name)
        # Before the component is added, we should not be able to find it
        assert not hasattr(model.components, name)
        # Add the element to a model and try to access it as attribute and item
        model.add_component(c)
        assert_equal(model.components[name], c)
        assert_equal(getattr(model.components, name), c)


@with_model
def test_component_name_existing_attribute():
    for name in ('_map', 'keys'):
        c = Monomer(name, _export=False)
        model.add_component(c)
        # When using an existing attribute name like_map, we should get able to
        # get it as an item, but not as an attribute
        assert_equal(model.components[name], c)
        assert_not_equal(getattr(model.components, name), c)


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
def test_monomer_rename_self_exporter():
    Monomer('A').rename('B')
    assert 'B' in model.monomers.keys()
    assert 'A' not in model.monomers.keys()


@with_model
def test_monomer_rename_non_self_exported_component():
    mon = Monomer('A', _export=False)
    mon.rename('B')
    assert mon.name == 'B'
    model.add_component(mon)
    mon.rename('C')
    assert mon.name == 'C'
    assert model.monomers['C'] == mon
    assert model.monomers.keys() == ['C']


def test_monomer_rename_non_self_exported_model():
    model = Model(_export=False)
    mon = Monomer('A', _export=False)
    model.add_component(mon)
    mon.rename('B')
    assert model.monomers.keys() == ['B']
    assert model.monomers['B'] == mon


@with_model
def test_invalid_state():
    Monomer('A', ['a', 'b'], {'a': ['a1', 'a2'], 'b': ['b1']})
    # Specify invalid state in Monomer.__call__
    assert_raises(ValueError, A, a='spam')
    # Specify invalid state in MonomerPattern.__call__
    assert_raises(ValueError, A(a='a1'), b='spam')


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
def test_duplicate_monomer_error():
    A = Monomer('A', ['a'])
    assert_raises(DuplicateMonomerError, (A(a=1) % A(a=1)), a=2)

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


@with_model
def test_complex_pattern_equivalence_state():
    """Ensure CP equivalence handles site states."""
    Monomer('A', ['s', 't'], {'t': ['x', 'y', 'z']})
    cp0 = A(s=1, t='x') % A(s=1, t='y')
    cp1 = A(s=1, t='y') % A(s=1, t='x')
    cp2 = A(s=1, t='x') % A(s=1, t='z')
    cp3 = A(s=1, t='z') % A(s=1, t='y')
    _check_pattern_equivalence((cp0, cp1))
    _check_pattern_equivalence((cp0, cp2), False)
    _check_pattern_equivalence((cp0, cp3), False)


@with_model
def test_complex_pattern_equivalence_bond_state():
    """Ensure CP equivalence handles bond and state on the same site."""
    Monomer('A', ['s'], {'s': ['x', 'y', 'z']})
    cp0 = A(s=('x', 1)) % A(s=('y', 1))
    cp1 = A(s=('y', 1)) % A(s=('x', 1))
    cp2 = A(s=('z', 1)) % A(s=('y', 1))
    cp3 = A(s='x') % A(s='y')
    _check_pattern_equivalence((cp0, cp1))
    _check_pattern_equivalence((cp0, cp2), False)
    _check_pattern_equivalence((cp0, cp3), False)


@with_model
def test_complex_pattern_equivalence_bond_numbering():
    """Ensure CP equivalence is insensitive to bond numbers."""
    Monomer('A', ['s'])
    cp0 = A(s=1) % A(s=1)
    cp1 = A(s=2) % A(s=2)
    _check_pattern_equivalence((cp0, cp1))


@with_model
def test_complex_pattern_equivalence_monomer_pattern_ordering():
    """Ensure CP equivalence is insensitive to MP order."""
    Monomer('A', ['s1', 's2'])
    cp0 = A(s1=1, s2=2) % A(s1=2, s2=1)
    cp1 = A(s1=2, s2=1) % A(s1=1, s2=2)
    _check_pattern_equivalence((cp0, cp1))


@with_model
def test_complex_pattern_equivalence_compartments():
    """Ensure CP equivalence is insensitive to Compartment"""
    Monomer('A', ['s1'])
    Compartment('C')
    cp0 = (A(s1=1) % A(s1=1)) ** C
    cp1 = (A(s1=1) ** C) % (A(s1=1) ** C)
    _check_pattern_equivalence((cp0, cp1))


@with_model
def test_reaction_pattern_match_complex_pattern_ordering():
    """Ensure CP equivalence is insensitive to MP order."""
    Monomer('A', ['s1', 's2'])
    cp0 = A(s1=1, s2=2) % A(s1=2, s2=1)
    cp1 = A(s1=2, s2=1) % A(s1=1, s2=2)
    rp0 = cp0 + cp1
    rp1 = cp1 + cp0
    rp2 = cp0 + cp0
    assert rp0.matches(rp1)
    assert rp1.matches(rp0)
    assert rp2.matches(rp0)


@with_model
def test_concreteness():
    Monomer('A', ['s'], {'s': ['x']})
    assert not (A(s=1) % A(s=1)).is_concrete()
    assert not (A(s=('x', 1)) % A(s=1)).is_concrete()
    assert (A(s=('x', 1)) % A(s=('x', 1))).is_concrete()
    assert (A(s='x')).is_concrete()
    assert not A().is_concrete()
    assert not A(s=None).is_concrete()
    assert not A(s=('x', ANY)).is_concrete()
    assert not A(s=WILD).is_concrete()

    Monomer('B', ['s'])
    assert not B().is_concrete()
    assert not B(s=ANY).is_concrete()
    assert B(s=1).is_concrete()

    Monomer('C')
    assert C().is_concrete()

    # Tests with compartments
    Compartment('cell')
    assert not C().is_concrete()
    assert (C() ** cell).is_concrete()


@with_model
def test_dangling_bond():
    Monomer('A', ['a'])
    Parameter('kf', 1.0)
    assert_raises(DanglingBondError, as_reaction_pattern, A(a=1) % A(a=None))


@with_model
def test_invalid_site_name():
    assert_raises(ValueError, Monomer, 'A', ['1'])


@with_model
def test_invalid_state_value():
    assert_raises(ValueError, Monomer, 'A', ['a'], {'a': ['1', 'a']})


@with_model
def test_valid_state_values():
    Monomer('A', ['a'], {'a': ['_1', '_b', '_', '_a', 'a']})


@with_model
def test_expression_type():
    assert_raises(ValueError, Expression, 'A', 1)


@with_model
def test_synth_requires_concrete():
    Monomer('A', ['s'], {'s': ['a', 'b']})
    Parameter('kA', 1.0)

    # These synthesis products are not concrete (site "s" not specified),
    # so they should raise a ValueError
    assert_raises(ValueError, Rule, 'r1', None >> A(), kA)
    assert_raises(ValueError, Rule, 'r2', A() | None, kA, kA)


@with_model
def test_rulepattern_match_none_against_state():
    Monomer('A', ['phospho'], {'phospho': ['u', 'p']})

    # A(phospho=None) should match unbound A regardless of phospho state,
    # so this should be a valid rule pattern
    A(phospho=None) + A(phospho=None) >> A(phospho=1) % A(phospho=1)

