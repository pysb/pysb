from PyJacobian import PyJacobian
from pysb.jacobian import JacobianGenerator
import sys, os, shutil, random, numpy, pylab

sys.path.append('models')
from earm_1_0 import model

def output_handler(icode, message):
    pass

def input_handler(icode, message):
    raise Exception("Jacobian backend asked for console input unexpectedly")

workdir = 'tmp_pysb_jac_%d_%d/' % (os.getpid(), random.randint(0, 10000))
os.mkdir(workdir)

jac_filename = workdir + 'model.jac'
gen = JacobianGenerator(model)
jac_file = file(jac_filename, 'w')
jac_file.write(gen.get_content(sim_length=72000))
jac_file.close()

pj = PyJacobian()
pj.setJacobianDirectory(all=workdir)
pj.addOutputCallback(output_handler)
pj.addInputCallback(input_handler)
pj.loadFile(jac_file.name)
pj.execute('SIM')
ts = pj.createTimeSeriesData()
pj.terminate()

full = numpy.memmap(filename=ts.binfile, dtype='float32', mode='r', offset=4, shape=(ts.getNumData(), ts.getNumVariables()), order='C')
indices = [ts._getIndex(name) for name in ('time', 'sim.m.tBid', 'sim.m.CPARP', 'sim.m.cSmac')]
a = full[:,indices]

last_i = numpy.searchsorted(a[:,0], 25000)
b = a[0:last_i,:]
t = b[:,0]
y = b[:,1:]
pylab.plot(t, y / y.max(0))
pylab.legend(('tBid', 'CPARP', 'cSmac'), loc='upper left', bbox_to_anchor=(0,1)).draw_frame(False)
pylab.show()

shutil.rmtree(workdir)
