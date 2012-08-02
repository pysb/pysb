import pysb.bng
import numpy
from scipy.integrate import ode
from scipy.weave import inline
import distutils.errors
import sympy
import re


use_inline = False
# try to inline a C statement to see if inline is functional
try:
    inline('int i;', force=1)
    use_inline = True
except distutils.errors.CompileError as e:
    pass

# try to load our pysundials cvode wrapper for scipy.integrate.ode
try:
    import pysb.pysundials_helpers
except ImportError as e:
    pass

# some sane default options for a few well-known integrators
default_integrator_options = {
    'vode': {
        'method': 'bdf',
        'with_jacobian': True,
        },
    'cvode': {
        'method': 'bdf',
        'iteration': 'newton',
        },
    }

def odesolve(model, t, param_values=None, y0=None, integrator='vode',
             **integrator_options):
    pysb.bng.generate_equations(model)
    
    if param_values is not None:
        # accept vector of parameter values as an argument
        if len(param_values) != len(model.parameters):
            raise Exception("param_values must be the same length as model.parameters")
        if not isinstance(param_values, numpy.ndarray):
            param_values = numpy.array(param_values)
    else:
        # create parameter vector from the values in the model
        param_subs = dict([ (p.name, p.value) for p in model.parameters ])
        param_values = numpy.array([param_subs[p.name] for p in model.parameters])

    code_eqs = '\n'.join(['ydot[%d] = %s;' % (i, sympy.ccode(model.odes[i])) for i in range(len(model.odes))])
    code_eqs = re.sub(r's(\d+)', lambda m: 'y[%s]' % (int(m.group(1))), code_eqs)
    for i, p in enumerate(model.parameters):
        code_eqs = re.sub(r'\b(%s)\b' % p.name, 'p[%d]' % i, code_eqs)

    # If we can't use weave.inline to run the C code, compile it as Python code instead for use with
    # exec. Note: C code with array indexing, basic math operations, and pow() just happens to also
    # be valid Python.  If the equations ever have more complex things in them, this might fail.
    if not use_inline:
        code_eqs_py = compile(code_eqs, '<%s odes>' % model.name, 'exec')

    if y0 is not None:
        # accept vector of species amounts as an argument
        if len(y0) != len(model.species):
            raise Exception("y0 must be the same length as model.species")
        if not isinstance(y0, numpy.ndarray):
            y0 = numpy.array(y0)
    else:
        y0 = numpy.zeros((len(model.odes),))
        for cp, ic_param in model.initial_conditions:
            pi = model.parameters.index(ic_param)
            si = model.get_species_index(cp)
            y0[si] = param_values[pi]

    def rhs(t, y, p):
        ydot = numpy.empty_like(y)
        # note that the evaluated code sets ydot as a side effect
        if use_inline:
            inline(code_eqs, ['ydot', 't', 'y', 'p']);
        else:
            exec code_eqs_py in locals()
        return ydot

    nspecies = len(model.species)
    obs_names = model.observables.keys()
    rec_names = ['__s%d' % i for i in range(nspecies)] + obs_names
    yout = numpy.ndarray((len(t), len(rec_names)))

    # build integrator options list from our defaults and any kwargs passed to this function
    options = {}
    try:
        options.update(default_integrator_options[integrator])
    except KeyError as e:
        pass
    options.update(integrator_options)

    # perform the actual integration
    integrator = ode(rhs).set_integrator(integrator, **options)
    integrator.set_initial_value(y0, t[0]).set_f_params(param_values)
    yout[0, :nspecies] = y0
    i = 1
    while integrator.successful() and integrator.t < t[-1]:
        integrator.integrate(t[i])
        yout[i, :nspecies] = integrator.y
        i += 1

    for i, obs in enumerate(model.observables):
        if obs.species:
            obs_values = (yout[:, obs.species] * obs.coefficients).sum(1)
        else:
            obs_values = numpy.zeros(yout.shape[0])
        yout[:, nspecies + i] = obs_values

    dtype = zip(rec_names, (yout.dtype,) * len(rec_names))
    yrec = numpy.recarray((yout.shape[0],), dtype=dtype, buf=yout)
    return yrec
