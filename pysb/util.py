from __future__ import print_function as _
from pysb import ComponentSet
import pysb.core
import inspect
import numpy
import io

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
    if not isinstance(parameter, pysb.core.Parameter):
        # Try getting the parameter by name
        parameter = model.parameters.get(parameter)
    cset = ComponentSet()
    for rule in model.rules:
        if rule.rate_forward is parameter or rule.rate_reverse is parameter:
            cset.add(rule)
    return cset


def synthetic_data(model, tspan, obs_list=None, sigma=0.1):
    #from pysb.integrate import odesolve
    from pysb.integrate import Solver
    solver = Solver(model, tspan)
    solver.run()

    # Sample from a normal distribution with variance sigma and mean 1
    # (randn generates a matrix of random numbers sampled from a normal
    # distribution with mean 0 and variance 1)
    # 
    # Note: This modifies yobs_view (the view on yobs) so that the changes 
    # are reflected in yobs (which is returned by the function). Since a new
    # Solver object is constructed for each function invocation this does not
    # cause problems in this case.
    solver.yobs_view *= ((numpy.random.randn(*solver.yobs_view.shape) * sigma) + 1)
    return solver.yobs


def get_param_num(model, name):
    for i in range(len(model.parameters)):
        if model.parameters[i].name == name:
            print(i, model.parameters[i])
            break
    return i


def write_params(model,paramarr, name=None):
    """ write the parameters and values to a csv file
    model: a model object
    name: a string with the name for the file, or None to return the content
    """
    if name is not None:
        fobj = open(name, 'w')
    else:
        fobj = io.StringIO()
    for i in range(len(model.parameters)):
        fobj.write("%s, %.17g\n"%(model.parameters[i].name, paramarr[i]))
    if name is None:
        return fobj.getvalue()


def update_param_vals(model, newvals):
    """update the values of model parameters with the values from a dict. 
    the keys in the dict must match the parameter names
    """
    update = []
    noupdate = []
    for i in model.parameters:
        if i.name in newvals:
            i.value = newvals[i.name]
            update.append(i.name)
        else:
            noupdate.append(i.name)
    return update, noupdate


def load_params(fname):
    """load the parameter values from a csv file, return them as dict.
    """
    parmsff = {}
    # FIXME: This might fail if a parameter name is larger than 50 characters.
    # FIXME: Maybe do this with the csv module instead?
    temparr = numpy.loadtxt(fname, dtype=([('a','S50'),('b','f8')]), delimiter=',') 
    for i in temparr:
        parmsff[i[0]] = i[1]
    return parmsff
