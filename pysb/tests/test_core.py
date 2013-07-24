from pysb.testing import *
from pysb.core import *
from functools import partial

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
