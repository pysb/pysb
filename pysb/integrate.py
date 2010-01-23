import pysb.bng
import numpy
from scipy.integrate import odeint
from scipy.weave import inline
import sympy
import re


def odesolve(model, t):
    pysb.bng.generate_equations(model)
    
    # FIXME code outside of model shouldn't have to handle parameter_overrides (same for initial_conditions below)
    param_subs = dict([ (sympy.Symbol(p.name), p.value) for p in model.parameters + model.parameter_overrides.values() ])

    c_code_consts = '\n'.join(['float %s = %e;' % (p.name, p.value) for p in model.parameters])
    c_code_eqs = '\n'.join(['ydot[%d] = %s;' % (i, sympy.ccode(model.odes[i])) for i in range(len(model.odes))])
    c_code_eqs = re.sub(r's(\d+)', lambda m: 'y[%s]' % (int(m.group(1))), c_code_eqs)
    c_code = c_code_consts + '\n\n' + c_code_eqs

    y0 = numpy.zeros((len(model.odes),))
    for cp, ic_param in model.initial_conditions:
        override = model.parameter_overrides.get(ic_param.name)
        if override is not None:
            ic_param = override
        si = model.get_species_index(cp)
        y0[si] = ic_param.value

    def rhs(y, t):
        ydot = y.copy()  # seems to be the fastest way to get an array of the same size?
        inline(c_code, ['y', 'ydot']); # sets ydot as a side effect
        return ydot

    nspecies = len(model.species)
    obs_names = [name for name, rp in model.observable_patterns]
    rec_names = ['__s%d' % i for i in range(nspecies)] + obs_names
    yout = numpy.ndarray((len(t), len(rec_names)))

    # perform the actual integration
    yout[:, :nspecies] = odeint(rhs, y0, t)

    for i, name in enumerate(obs_names):
        factors, species = zip(*model.observable_groups[name])
        yout[:, nspecies + i] = (yout[:, species] * factors).sum(1)

    dtype = zip(rec_names, (yout.dtype,) * len(rec_names))
    yrec = numpy.recarray((yout.shape[0],), dtype=dtype, buf=yout)
    return yrec
