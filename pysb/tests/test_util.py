from pysb.core import Initial, Model, Monomer, Parameter, Rule
from pysb.testing import with_model
from pysb.util import alias_model_components, rules_using_parameter


def test_alias_model_components():
    """
    Tests that alias_model_components() exports to the global namespace
    """
    m = Model(_export=False)
    m.add_component(Monomer('A', _export=False))
    assert 'A' not in globals()
    alias_model_components(m)

    # A should now be defined in the namespace - try deleting it
    assert isinstance(globals()['A'], Monomer)

    # Delete the monomer to cleanup the global namespace
    del globals()['A']


@with_model
def test_rules_using_parameter():
    """
    Tests for rules_using_parameter() in pysb.util
    """
    Monomer('m1')
    Monomer('m2')
    Monomer('m3')

    ka1 = Parameter('ka1', 2e-5)
    keff = Parameter('keff', 1e5)

    Initial(m2(), Parameter('m2_0', 10000))

    Rule('R1', None >> m1(), ka1)
    Rule('R2', m1() + m2() >> m1() + m3(), keff)

    components = rules_using_parameter(model, 'keff')
    assert R2 in components

    # Get rules by supplying Parameter object directly
    components = rules_using_parameter(model, keff)
    assert R2 in components
