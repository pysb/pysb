from pysb.testing import *
from pysb.core import *
from functools import partial
from copy import deepcopy
from pysb.core import MonomerPattern

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
    
    o=Monomer('B', ['x','y'], _export=False)
    ok_(m != o)
    assert_equal(m,m)
    o = deepcopy(m)
    assert_equal(m, o)
    o.sites=['y','x', 'z']
    assert_equal(m, o)

def test_monomer_pattern():
    sites = ['x', 'y', 'z']
    states = {'y': ['foo', 'bar', 'baz'], 'x': ['e']}
    m = Monomer('A', sites, states, _export=False)
    mp = MonomerPattern(m, {'x':ANY}, Compartment("Joe", _export=False))
    o  = deepcopy(mp)
    assert_equal(mp, o)
    o.compartment = Compartment("Jim", _export=False)
    ok_(mp != o)
    o  = deepcopy(mp)
    o.site_conditions = {'x':'e'}
    ok_(mp != o)
    o  = deepcopy(mp)
    mp.monomer = Monomer('B', sites, states, _export=False)
    ok_(mp != o)

def test_compartment():
    c = Compartment("Joe", _export=False)
    o = deepcopy(c)
    ok_(c == o)
    o.name = "Jim"
    ok_(c != o)
    o = deepcopy(c)
    o.size = 2
    ok_(c != o)

def test_parameter():
    p = Parameter("a", 2.3, _export=False)
    o = deepcopy(p)
    ok_(o==p)
    o.value = 2.3
    ok_(o!=p)
    o = deepcopy(p)
    o.name = "b"
    ok_(o!=p)

@with_model
def test_monomer_model():
    Monomer('A', ['x','y'])
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
    Monomer('A')
    Monomer('B', ['x', 'y'], {'x': ['e', 'f']})
    Parameter('k', 1.0)
    Rule('bind', A() + B(x='e', y=WILD) >> A() % B(x='f', y=None), k, k)
    model2 = pickle.loads(pickle.dumps(model))
    check_model_against_component_list(model, model2.all_components())

@with_model
def test_compartment_initial_error():
    Monomer('A', ['s'])
    Parameter('A_0', 2.0)
    c1 = Compartment("C1")
    c2 = Compartment("C2")
    Initial(A(s=None)**c1, A_0)
    Initial(A(s=None)**c2, A_0)
    
