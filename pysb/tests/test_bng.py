from pysb.testing import *
from pysb import *
from pysb.bng import *
from pysb.core import as_complex_pattern

@with_model
def test_generate_network():
    Monomer('A')
    assert_raises(NoInitialConditionsError, generate_network, model)
    Parameter('A_0', 1)
    Initial(A(), A_0)
    assert_raises(NoRulesError, generate_network, model)
    Parameter('k', 1)
    Rule('degrade', A() >> None, k)
    ok_(generate_network(model))

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
