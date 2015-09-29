from pysb.core import *
from pysb.annotation import Annotation

__all__ = ['Observable', 'Initial', 'MatchOnce', 'Model', 'Monomer',
           'Parameter', 'Compartment', 'Rule', 'Expression', 'ANY', 'WILD',
           'Annotation']

try:
    import reinteract         # fails if reinteract not installed
    reinteract.custom_result  # fails if this code is run outside of the reinteract shell
except (ImportError, AttributeError) as e:
    pass                      # silently skip applying the mixin below
else:
    import pysb.reinteract_integration
    pysb.reinteract_integration.apply_mixins()

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
