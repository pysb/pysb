from pysb import ComponentSet
import pysb.core
import inspect
import numpy
from pysb.integrate import odesolve

__all__ = ['alias_model_components', 'rules_using_parameter']

def alias_model_components(model=None):
    """Make all model components visible as symbols in the caller's global namespace"""
    if model is None:
        model = pysb.core.SelfExporter.default_model
    caller_globals = inspect.currentframe().f_back.f_globals
    components = dict((c.name, c) for c in model.all_components())
    caller_globals.update(components)

def rules_using_parameter(model, parameter):
    """Return a ComponentSet of rules in the model which make use of the given parameter"""
    cset = ComponentSet()
    for rule in model.rules:
        if rule.rate_forward is parameter or rule.rate_reverse is parameter:
            cset.add(rule)
    return cset


def synthetic_data(model, tspan, obs_list=None, sigma=0.1, seed=None):
    random = numpy.random.RandomState(seed)
    ysim = odesolve(model, tspan)
    ysim_array = ysim.view().reshape(len(tspan), len(ysim.dtype))
    ysim_array *= (random.randn(*ysim_array.shape) * sigma + 1);
    return ysim

