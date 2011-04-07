from PyJacobian import PyJacobian
from pysb.jacobian import JacobianGenerator
import os

os.chdir('models')
from earm_1_0 import model

workdir = '/tmp/%d_%d/' % (os.getpid(), random.randint(0, 10000))
print "working dir:" + workdir
jac_filename = workdir + 'model.jac'

gen = JacobianGenerator(model)
jac_file = file(jac_filename, 'w')
jac_file.write(gen.get_content)
jac_file.close()

pj = PyJacobian()
pj.load(jac_file)
pj.execute('SIM')
