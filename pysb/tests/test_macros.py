from pysb.testing import *
from pysb.core import *
from pysb.macros import *

@with_model
def test_catalyze_one_step_None():
    # What if no substrate is required, hand-wavy, but present in published models!
    Monomer('E', ['b'])
    Monomer('P')
    catalyze_one_step(E, None, P, 1e-4)
    assert_equal(len(model.rules), 1)