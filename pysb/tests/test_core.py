from pysb.testing import *
from pysb.core import *
import pytest
import operator
import re


@with_model
def test_component_names_valid():
    for name in 'a', 'B', 'AbC', 'dEf', '_', '_7', '__a01b__999x_x___':
        c = Monomer(name, _export=False)
        assert c.name == name
        # Before the component is added, we should not be able to find it
        assert not hasattr(model.components, name)
        # Add the element to a model and try to access it as attribute and item
        model.add_component(c)
        assert model.components[name] == c
        assert getattr(model.components, name) == c

@with_model
def test_deepcopy_parameter():
    Parameter("a", 1)
    copy.deepcopy(model)


@with_model
def test_component_name_existing_attribute():
    for name in ('_map', 'keys'):
        c = Monomer(name, _export=False)
        model.add_component(c)
        # When using an existing attribute name like_map, we should get able to
        # get it as an item, but not as an attribute
        assert model.components[name] == c
        assert getattr(model.components, name) != c


def test_component_names_invalid():
    for name in 'a!', '!B', 'A!bC~`\\', '_!', '_!7', '__a01b  999x_x___!':
        with pytest.raises(InvalidComponentNameError):
            Component(name, _export=False)

@with_model
def test_function_introspection():
    # Case 1: Component defined inside function
    Monomer('A')
    assert A._function == 'test_function_introspection'

    # Case 2: Component defined inside nested function
    def define_monomer_b():
        Monomer('B')
    define_monomer_b()
    assert B._function == 'define_monomer_b'

    # Case 3: Component defined by macro
    from pysb.macros import equilibrate
    equilibrate(A(), B(), [1, 1])

    assert model.rules['equilibrate_A_to_B']._function == 'equilibrate'

    # Case 4: Component defined by macro inside function
    def define_macro_inside_function():
        Monomer('C')
        equilibrate(A(), C(), [2, 2])
    define_macro_inside_function()
    assert model.rules['equilibrate_A_to_C']._function == 'equilibrate'


def test_monomer():
    sites = ['x', 'y', 'z']
    states = {'y': ['foo', 'bar', 'baz'], 'x': ['e']}
    m = Monomer('A', sites, states, _export=False)
    assert m.sites == sites
    assert m.site_states == states
    assert type(m()) == MonomerPattern

    with pytest.raises(ValueError):
        Monomer('A', 'x', _export=False)
    with pytest.raises(Exception):
        Monomer('A', 'x', 'x', _export=False)
    with pytest.raises(Exception):
        Monomer('A', ['x'], {'y': ['a']}, _export=False)
    with pytest.raises(Exception):
        Monomer('A', ['x'], {'x': [1]}, _export=False)

@with_model
def test_monomer_model():
    Monomer('A')
    assert A in model.monomers
    assert A in model.all_components()
    assert A not in model.all_components() - model.monomers


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
    with pytest.raises(ValueError):
        A(a='spam')
    # Specify invalid state in MonomerPattern.__call__
    with pytest.raises(ValueError):
        A(a='a1')(b='spam')


@with_model
def test_initial():
    Monomer('A', ['s'])
    Parameter('A_0')
    Initial(A(s=None), A_0)
    for obj, args in (
        ('not a complexpattern', A_0),
        (A(), A_0),
        (A(s=None), A_0),
        (MatchOnce(A(s=None)), A_0),
    ):
        with pytest.raises(InvalidInitialConditionError):
            Initial(*args)

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
    assert len(model.all_components()) == 7
    model2 = pickle.loads(pickle.dumps(model))
    check_model_against_component_list(model, model2.all_components())

@with_model
def test_model_repr():
    monomers = [Monomer(f"A{i}") for i in range(1, 7)]
    parameters = [Parameter(f"P{i}") for i in range(1, 5)]
    rules = [
        Rule(f"R{i}", m() >> None, parameters[0])
        for i, m in enumerate(monomers[:5], 1)
    ]
    expressions = [
        Expression(f"E{i}", p + 1)
        for i, p in enumerate(parameters[:3], 1)
    ]
    Compartment("C1")
    Compartment("C2", C1, 2)
    EnergyPattern("G1", A1() % A2(), P1)
    assert re.match(
        r"<Model 'test_model_repr' \(monomers: 6, rules: 5, parameters: 4, "
        r"expressions: 3, compartments: 2, energypatterns: 1\) at 0x[0-9a-f]+>",
        repr(model)
    )

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
def test_expand_obs_no_coeffs():
    A = Monomer('A')
    o = Observable('A_obs', A())
    # Test that expand_obs works with observable when no coefficients or
    # species are present
    o.expand_obs()

@with_model
def test_compartment_initial_error():
    Monomer('A', ['s'])
    Parameter('A_0', 2.0)
    c1 = Compartment("C1")
    c2 = Compartment("C2")
    Initial(A(s=None)**c1, A_0)
    Initial(A(s=None)**c2, A_0)

@with_model
def test_compartment():
    # Ensure that compartment size can be specified by a parameter,
    #  a constant expression, or be omitted.
    Parameter('A', 1.0)
    Parameter('B', 2.0)
    Expression('E', A * B)
    Compartment("C1")
    Compartment("C2", C1, 2, A)
    Compartment("C3", C1, 2, E)

@with_model
def test_monomer_pattern_add_to_none():
    """Ensure that MonomerPattern + None returns a ReactionPattern."""
    Monomer('A', ['s'])
    assert isinstance(A() + None, ReactionPattern), 'A() + None did not return a ReactionPattern.'

@with_model
def test_complex_pattern_add_to_none():
    """Ensure that ComplexPattern + None returns a ReactionPattern."""
    Monomer('A', ['s'])
    assert isinstance(A(s=1) % A(s=1) + None, ReactionPattern), 'A(s=1) % A(s=1) + None did not return a ReactionPattern.'

@with_model
def test_reaction_pattern_add_to_none():
    """Ensure that ReactionPattern + None returns a ReactionPattern."""
    Monomer('A', ['s'])
    cp = A(s=1) % A(s=1)
    rp = cp + cp
    assert isinstance(rp + None, ReactionPattern), 'ReactionPattern + None did not return a ReactionPattern.'

@with_model
def test_complex_pattern_call():
    """Ensure ComplexPattern calling (refinement) works as expected."""
    Monomer('A', ['w', 'x'], {'x': ('e', 'f')})
    Monomer('B', ['y', 'z'], {'z': ('g', 'h')})
    cp = A(w=1, x='e') % B(y=1, z='g')
    r = {'x': 'f', 'z': 'h'} # refinement for complexpattern
    assert cp(**r)
    assert cp(**r).monomer_patterns[0].site_conditions['x'] == r['x']
    assert cp(r)
    assert cp(r).monomer_patterns[0].site_conditions['x'] == r['x']
    with pytest.raises(RedundantSiteConditionsError):
        cp({'x': 'f'}, z='h')

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
def test_complex_pattern_single_monomer_complex_compartment():
    Monomer('A')
    Compartment('C')
    Parameter('A_0', 10)
    cp_source = as_complex_pattern(A()) ** C

    # Check executing the string representation of the above ComplexPattern
    # leads to the correct application of compartment, i.e. compartment applies
    # to complex, not to monomer
    exec(f"Initial({str(cp_source)}, A_0)")

    cp_in_model = model.initials[0].pattern
    assert cp_in_model.compartment == C
    assert cp_in_model.monomer_patterns[0].compartment is None


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
    with pytest.raises(DanglingBondError):
        as_reaction_pattern(A(a=1) % A(a=None))


@with_model
def test_invalid_site_name():
    with pytest.raises(ValueError):
        Monomer('A', ['1'])


@with_model
def test_invalid_state_value():
    with pytest.raises(ValueError):
        Monomer('A', ['a'], {'a': ['1', 'a']})


@with_model
def test_valid_state_values():
    Monomer('A', ['a'], {'a': ['_1', '_b', '_', '_a', 'a']})


@with_model
def test_expression_type():
    with pytest.raises(ValueError):
        Expression('A', 1)


@with_model
def test_expression_evaluation():
    Parameter('k1', 10)
    Expression('k2', 2 * k1)
    Expression('k3', k2/2)
    assert int(k2.get_value()) == 20
    assert int(k3.get_value()) == 10


@with_model
def test_synth_requires_concrete():
    Monomer('A', ['s'], {'s': ['a', 'b']})
    Parameter('kA', 1.0)

    # These synthesis products are not concrete (site "s" not specified),
    # so they should raise a ValueError
    with pytest.raises(ValueError):
        Rule('r1', None >> A(), kA)
    with pytest.raises(ValueError):
        Rule('r2', A() | None, kA, kA)


@with_model
def test_rulepattern_match_none_against_state():
    Monomer('A', ['phospho'], {'phospho': ['u', 'p']})

    # A(phospho=None) should match unbound A regardless of phospho state,
    # so this should be a valid rule pattern
    A(phospho=None) + A(phospho=None) >> A(phospho=1) % A(phospho=1)


@with_model
def test_tags():
    Monomer('A', ['b'])
    Tag('x')

    # Use __matmul__ instead of @ for Python 2.7 support in tests
    assert repr(x) == "Tag('x')"
    assert repr(x.__matmul__(A()) % A()) == 'x @ A() % A()'
    assert repr((A() % A()).__matmul__(x)) == 'A() % A() @ x'

    # Postfix tags should auto-upgrade a MonomerPattern to a ComplexPattern
    assert isinstance(A().__matmul__(x), ComplexPattern)

    # Trying to extend a tagged complex should fail - the tag should always
    # be specified last
    with pytest.raises(ValueError):
        operator.mod((A(b=1) % A(b=1)).__matmul__(x),
                  A(b=1))

    Observable('o1', A(b=None))

    # Create an expression containing a tag
    Expression('e_no_tag', o1 ** 2)
    Expression('e_tag', o1(x) ** 2)

    # Test tag defined in rate but not in rule expression
    with pytest.raises(ValueError):
        Rule('r1', None >> A(b=None), e_tag)

    # Test tag defined in rule expression but not in rate
    Rule('r2', None >> A(b=None).__matmul__(x), e_no_tag)

    # Test tag with compartment
    Compartment('c')
    assert repr((A().__matmul__(x)) ** c) == 'as_complex_pattern(A()) ** c @ x'
    assert repr((A() ** c).__matmul__(x)) == 'A() ** c @ x'


@with_model
def test_multi_bonds():
    Monomer('A', ['a'])
    a_pat = A(a=[1, 2])

    # Check _as_graph() works for multi-bonds
    a_pat._as_graph()

    assert a_pat.is_concrete()


@with_model
def test_duplicate_sites():
    Monomer('A', ['a', 'a'])
    Monomer('B', ['b', 'b'], {'b': ['u', 'p']})

    assert not A(a=1).is_concrete()
    assert A(a=MultiState(1, 2)).is_concrete()
    assert A(a=MultiState(1, None)).is_concrete()

    assert not B(b=('u', 1)).is_concrete()

    assert B(b=MultiState('u', 'p')).is_concrete()
    assert B(b=MultiState(('u', 1), ('u', 2))).is_concrete()

    # Check _as_graph() works for duplicate sites
    B(b=MultiState(('u', 1), ('u', 2)))._as_graph()

    assert B(b=MultiState('u', ('u', 1))).is_concrete()

    # Syntax errors (should use MultiState)
    with pytest.raises(ValueError):
        B(b=('u', 'p'))
    with pytest.raises(ValueError):
        B(b=['u', 'p'])

    # Syntax error (can't nest MultiState)
    with pytest.raises(ValueError):
        MultiState(MultiState(1, 2), 'p')

    # Duplicate sites with multi-bond
    A(a=MultiState([1, 2], [1, 2]))


@pytest.mark.raises(exception=ValueError)
def test_duplicate_site_single_site():
    MultiState('a')


@with_model
def test_invalid_rule():
    Monomer('A')
    with pytest.raises(ExpressionError):
        Rule('r1', None >> A(), 1.0)
    assert len(model.rules) == 0

    Parameter('kf', 1.0)
    with pytest.raises(Exception):
        Rule('r1', 'invalid_rule_expr', kf)
    assert len(model.rules) == 0


@with_model
def test_invalid_expression():
    with pytest.raises(ValueError):
        Expression('e1', 'invalid_expr')
    assert len(model.expressions) == 0


@with_model
def test_invalid_monomer_name():
    with pytest.raises(ValueError):
        Monomer('a', 123)
    assert len(model.monomers) == 0


@with_model
def test_invalid_parameter():
    with pytest.raises(ValueError):
        Parameter('a', 'invalid_value')
    assert len(model.parameters) == 0


@with_model
def test_invalid_compartment():
    with pytest.raises(Exception):
        Compartment('c1', 'invalid_parent')

    # Invalid dynamic expression as compartment size
    Monomer('A')
    Observable('O', A)
    Expression('E', O)
    with pytest.raises(Exception):
        Compartment('c2', size=E)

    assert len(model.compartments) == 0


@with_model
def test_invalid_observable():
    assert_raises(InvalidReactionPatternException,
                  Observable, 'o1', 'invalid_pattern')
    assert len(model.observables) == 0


@with_model
def test_update_initial_condition():
    Monomer('A')
    Monomer('B')
    Parameter('k', 1.0)
    Initial(A(), k)

    model.update_initial_condition_pattern(A(), B())

    assert len(model.initials) == 1
    assert as_complex_pattern(B()).is_equivalent_to(
        as_complex_pattern(model.initials[0].pattern))


def test_model_not_defined():
    with pytest.raises(ModelNotDefinedError):
        Monomer('A')


@pytest.mark.raises(exception=ReusedBondError)
@with_model
def test_bind_multiple():
    Monomer('A', ['a'])
    Monomer('B', ['b'])

    as_reaction_pattern(A(a=1) % B(b=1) % B(b=1))


@pytest.mark.raises(exception=ValueError)
@with_model
def test_reverse_rate_non_reversible_rule():
    Monomer('A')
    Parameter('kf', 1)
    Parameter('kr', 2)
    Rule('r1', None >> A(), kf, kr)


@with_model
def test_parameter_assumptions():
    Parameter('k1', 0.0)
    assert k1.is_real
    assert k1.is_nonnegative
    assert not k1.is_integer
    Parameter('k2', 0.0, nonnegative=False)
    assert not k2.is_nonnegative
    Parameter('k3', 0.0, integer=True)
    assert k3.is_integer


@pytest.mark.raises(exception=ValueError)
@with_model
def test_parameter_noninteger_integer_init():
    Parameter('k3', 0.3, integer=True)


@pytest.mark.raises(exception=ValueError)
@with_model
def test_parameter_noninteger_integer_setter():
    Parameter('k3', 1.0, integer=True)
    k3.value = 0.4


@pytest.mark.raises(exception=ValueError)
@with_model
def test_parameter_negative_nonnegative_init():
    Parameter('k3', -0.2, nonnegative=True)


@pytest.mark.raises(exception=ValueError)
@with_model
def test_parameter_negative_nonnegative_setter():
    Parameter('k3', 0.0, nonnegative=True)
    k3.value = -0.2


@with_model
def test_reversible_synthesis():
    Monomer('A')
    Parameter('k', 1)
    Rule('r1', None | A(), k, k)
    Rule('r2', None | as_complex_pattern(A()), k, k)


@with_model
def test_energy():
    Monomer('A', ['a', 'b'])
    Monomer('B', ['a'])
    Parameter('RT', 2)
    Parameter('A_0', 10)
    Parameter('AB_0', 10)
    Parameter('phi', 0)
    Expression('E_AAB_RT', -5 / RT)
    Expression('E0_AA_RT', -1 / RT)
    Rule(
        'A_dimerize',
        A(a=None) + A(a=None) | A(a=1) % A(a=1),
        phi,
        E0_AA_RT,
        energy=True,
    )
    EnergyPattern('epAAB', A(a=1) % A(a=1, b=2) % B(a=2), E_AAB_RT)
    Initial(A(a=None, b=None), A_0)
    Initial(A(a=None, b=1) % B(a=1), AB_0)

    assert "energy=True" in repr(A_dimerize)
