from pysb import ComponentSet
import pysb.core
import inspect
import numpy

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
    from pysb.integrate import odesolve
    random = numpy.random.RandomState(seed)
    ysim = odesolve(model, tspan)
    ysim_array = ysim.view().reshape(len(tspan), len(ysim.dtype))
    ysim_array *= (random.randn(*ysim_array.shape) * sigma + 1);
    return ysim

def get_param_num(model, name):
    for i in range(len(model.parameters)):
        if model.parameters[i].name == name:
            print i, model.parameters[i]
            break
    return i


def write_params(model,paramarr, name):
    """ write the parameters and values to a csv file
    model: a model object
    name: a string with the name for the file
    
    """
    fobj = open(name, 'w')
    for i in range(len(model.parameters)):
        fobj.write("%s, %g\n"%(model.parameters[i].name, paramarr[i]))
    fobj.close()

def update_param_vals(model, nvals):
    """update the values of model parameters with the values
    from an array. 
    This assumes newvals and model.parameters are in the same order!!!
    """
    if len(model.parameters) == len(nvals):
        for i in range(len(newvals)):
            model.parameters[i].value = nvals[i]


def load_params(model, fname):
    """load the parameter values from a csv file
    the parameters should be stored as
    name, value entries in a csv-type file
    """
    parmsfromfile = numpy.loadtxt(fname, dtype=([('a','S50'),('b','f8')]), delimiter=',')
    return parmsfromfile
