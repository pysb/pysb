import pysb
import pysb.bng
import sympy
import re
import sys
import os


model_filename = sys.argv[1]
# sanity checks on filename
if not os.path.exists(model_filename):
    raise Exception("File '%s' doesn't exist" % model_filename)
if not re.search(r'\.py$', model_filename):
    raise Exception("File '%s' is not a .py file" % model_filename)
sys.path.append(os.path.dirname(model_filename))
model_name = re.sub(r'\.py$', '', os.path.basename(model_filename))
# import it
try:
    model_module = __import__(model_name)
except StandardError as e:
    print "Error in model script:\n"
    raise
# grab the 'model' variable from the module
try:
    model = model_module.__dict__['model']
except KeyError:
    raise Exception("File '%s' isn't a model file" % model_filename)

pysb.bng.generate_equations(model)

obs_names = [name for name, rp in model.observable_patterns]

ic_values = [0] * len(model.odes)
for cp, ic_param in model.initial_conditions:
    ic_values[model.get_species_index(cp)] = ic_param.value

# list of "dynamic variables"
pw_x = ["m = pwAddX(m, 's%d', %e);" % (i, ic_values[i]) for i in range(len(model.odes))]

# parameters
pw_k = ["m = pwAddK(m, '%s', %e);" % (p.name, p.value) for p in model.parameters]

# equations (one for each dynamic variable)
# Note that we just generate C code, which for basic math expressions
# is identical to matlab.  We just have to change 'pow' to 'power'.
# Ideally there would be a matlab formatter for sympy.
pw_ode = ["m = pwAddODE(m, 's%d', '%s');" % (i, sympy.ccode(model.odes[i])) for i in range(len(model.odes))]
pw_ode = [re.sub(r'pow(?=\()', 'power', s) for s in pw_ode]

# observables or "derived variables"
pw_z = ["m = pwAddZ(m, '%s', '%s');" % (' + '.join(['%f * s%s' % g for g in model.observable_groups[name]]), name) for name in obs_names]

print '% PottersWheel model definition file'
print 'function m = %s()' % (model_name)
print ''
print 'm = pwGetEmptyModel();'
print ''
print '% meta information'
print "m.ID          = '%s';" % model_name
print "m.name        = '%s';" % model_name
print "m.description = '';"
print "m.authors     = {''};"
print "m.dates       = {''};"
print "m.type        = 'PW-1-5';"
print ''
print '% dynamic variables'
print '\n'.join(pw_x)
print ''
print '% dynamic parameters'
print '\n'.join(pw_k)
print ''
print '% ODEs'
print '\n'.join(pw_ode)
print ''
print '% derived variables'
print '\n'.join(pw_z)
