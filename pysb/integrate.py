import pysb.bng
import numpy
from scipy.integrate import ode
from scipy.integrate.ode import IntegratorBase, vode
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

def odesolve(model, t, integrator='vode', **integrator_options):
    pysb.bng.generate_equations(model)
    
    param_subs = dict([ (p.name, p.value) for p in model.parameters ])
    param_values = numpy.array([param_subs[p.name] for p in model.parameters])
    param_indices = dict( (p.name, i) for i, p in enumerate(model.parameters) )

    code_eqs = '\n'.join(['ydot[%d] = %s;' % (i, sympy.ccode(model.odes[i])) for i in range(len(model.odes))])
    code_eqs = re.sub(r's(\d+)', lambda m: 'y[%s]' % (int(m.group(1))), code_eqs)
    for i, p in enumerate(model.parameters):
        code_eqs = re.sub(r'\b(%s)\b' % p.name, 'p[%d]' % i, code_eqs)

    # If we can't use weave.inline to run the C code, compile it as Python code instead for use with
    # exec. Note: C code with array indexing, basic math operations, and pow() just happens to also
    # be valid Python.  If the equations ever have more complex things in them, this might fail.
    if not use_inline:
        code_eqs_py = compile(code_eqs, '<%s odes>' % model.name, 'exec')

    y0 = numpy.zeros((len(model.odes),))
    for cp, ic_param in model.initial_conditions:
        si = model.get_species_index(cp)
        y0[si] = ic_param.value

    def rhs(t, y, p):
        ydot = numpy.empty_like(y)
        # note that the evaluated code sets ydot as a side effect
        if use_inline:
            inline(code_eqs, ['ydot', 't', 'y', 'p']);
        else:
            exec code_eqs_py in locals()
        return ydot

    nspecies = len(model.species)
    obs_names = [name for name, rp in model.observable_patterns]
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

    for i, name in enumerate(obs_names):
        factors, species = zip(*model.observable_groups[name])
        yout[:, nspecies + i] = (yout[:, species] * factors).sum(1)

    dtype = zip(rec_names, (yout.dtype,) * len(rec_names))
    yrec = numpy.recarray((yout.shape[0],), dtype=dtype, buf=yout)
    return yrec

try:
    from pysundials import cvode as _cvode, nvecserial as _nvecserial
except ImportError as e:
    _cvode = None

class cvode(IntegratorBase):
    if _cvode:
        valid_methods = {
            'adams': _cvode.CV_ADAMS,
            'bdf': _cvode.CV_BDF,
            }
        valid_iterations = {
            'functional': _cvode.CV_FUNCTIONAL,
            'newton': _cvode.CV_NEWTON,
            }

    def __init__(self, method='adams', iteration='functional', rtol=1.0e-7, atol=1.0e-11):
        if method not in cvode.valid_methods:
            raise Exception("%s is not a valid value for method -- please use one of the following: %s" %
                            (method, [m for m in cvode.valid_methods]))
        if iteration not in cvode.valid_iterations:
            raise Exception("%s is not a valid value for iteration -- please use one of the following: %s" %
                            (iteration, [m for m in cvode.valid_iterations]))
        self.method = method
        self.iteration = iteration
        self.rtol = rtol
        self.atol = atol
        self.t0 = 0.0
        self.y0 = None
        self.first_step = True

    def reset(self, n, has_jac):
        if has_jac:
            raise Exception("has_jac not supported")
        self.y0 = numpy.empty(n)
        self.first_step = True
        # initialize the cvode memory object
        self.cvode_mem = _cvode.CVodeCreate(cvode.valid_methods[self.method],
                                            cvode.valid_iterations[self.iteration])

    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        if self.first_step:
            def cvode_f(t, y, ydot, f_data):
                ydot[:] = f(t, y, f_data)
            # allocate memory for cvode
            cvode.CVodeMalloc(self.cvode_mem, cvode_f, t0, _cvode.NVector(self.y0),
                              _cvode.CV_SS, self.rtol, self.atol)
            # link integrator with linear solver
            cvodes.CVDense(cvodes_mem, len(self.y0))

if _cvode:
    IntegratorBase.integrator_classes.append(cvode)
