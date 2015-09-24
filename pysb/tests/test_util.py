from pysb.core import *
from pysb.util import *
from pysb.examples import earm_1_0


def test_alias_model_components():
    """
    Tests that alias_model_components() exports to the global namespace
    """
    m = Model(_export=False)
    m.add_component(Monomer('temp_monomer_A', _export=False))
    try:
        temp_monomer_A
        raise NameError('Monomer A should not be defined in global namespace')
    except NameError:
        # We expect this exception due to _export=False
        pass
    alias_model_components(m)

    # Use global version of temp_monomer_A (needed to delete it)
    global temp_monomer_A

    # A should now be defined in the namespace - try deleting it
    assert isinstance(temp_monomer_A, Monomer)

    # Delete the monomer to cleanup the global namespace
    del temp_monomer_A


def test_rules_using_parameter():
    """
    Tests for rules_using_parameter() in pysb.util
    """
    # Get rules matching parameter name
    r = rules_using_parameter(earm_1_0.model, 'kc1')
    assert isinstance(r.get('produce_DISC'), Rule)

    # Get rules by supplying Parameter object directly
    r = rules_using_parameter(earm_1_0.model,
                              earm_1_0.model.parameters.get('kc1'))
    assert isinstance(r.get('produce_DISC'), Rule)