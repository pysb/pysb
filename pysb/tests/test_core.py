from pysb.testing import *
from pysb.core import *

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
