from pysb.core import *

__all__ = ['Observe', 'Initial', 'MatchOnce', 'Model', 'Monomer',
           'Parameter', 'Compartment', 'Rule', 'ANY', 'WILD']

try:
    import reinteract         # fails if reinteract not installed
    reinteract.custom_result  # fails if this code is run outside of the reinteract shell
except (ImportError, AttributeError) as e:
    pass                      # silently skip applying the mixin below
else:
    import pysb.reinteract_integration
    pysb.reinteract_integration.apply_mixins()
