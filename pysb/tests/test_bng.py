from pysb.testing import *
from pysb import *
from pysb.bng import *

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
