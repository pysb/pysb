import pysb
import pysb.bng
import sympy
import re
import sys


model_name = sys.argv[1]

exec("from %s import model" % (model_name));


pysb.bng.generate_equations(model)

obs_names = [name for name, rp in model.observable_patterns]

pw_x = '\n'.join(["m = pwAddX(m, 's%d', 0);" % (i) for i in range(len(model.odes))])
pw_k = '\n'.join(["m = pwAddK(m, '%s', %e);" % (p.name, p.value) for p in model.parameters])
pw_ode = '\n'.join(["m = pwAddODE(m, 's%d', '%s');" % (i, sympy.ccode(model.odes[i])) for i in range(len(model.odes))])
pw_ode = re.sub(r'pow(?=\()', 'power', pw_ode)

pw_z = '\n'.join(["m = pwAddZ(m, '%s', '%s');" % (' + '.join(['%f * s%s' % g for g in model.observable_groups[name]]), name) for name in obs_names])

print 'function m = %s()' % (model_name)
print ''
print 'm = pwGetEmptyModel();'
print ''
print pw_x
print ''
print pw_k
print ''
print pw_ode
print ''
print pw_z
