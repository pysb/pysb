import pysb.core
from pysb.generator.bng import BngGenerator
import os
import subprocess
import random
import re
import itertools
import sympy
import numpy
from StringIO import StringIO
import pysb.bng as BNG
import numpy as np

def run(model,tspan,n_runs,param_values=None,verbose=False,**additional_args):
    output_dir=os.getcwd()
    working_dir = os.getcwd()
    os.chdir(output_dir)
    for i in xrange(n_runs):
        print i
        generate_bng_file(model, tspan)
    trajectories = gather_data(model,output_dir)
    return trajectories
def generate_bng_file(model,tspan,param_values=None,**additional_args):
    Seed = np.random.randint(0,100000)
    output_file_basename = None
    ssa_args = "t_start=>%s,sample_times=>%s" % (str(tspan[0]),str(list(tspan)))
    for key,val in additional_args.items(): ssa_args += ", %s=>%s" % (key,"\""+str(val)+"\"" if isinstance(val,str) else str(val))
    ssa_args += ",seed=>%s" % str(Seed)
    run_ssa_code = """
begin actions
    generate_network({overwrite=>1})
    simulate_ssa({%s})
end actions
""" % (ssa_args)
    if param_values is not None:
        if len(param_values) != len(model.parameters):
            raise Exception("param_values must be the same length as model.parameters")
        for i in range(len(param_values)):
            model.parameters[i].value = param_values[i]
    
    gen = BngGenerator(model)
    if output_file_basename is None:
        output_file_basename = '%s_%d_%d_temp' % (model.name,
                                os.getpid(), Seed)

    if os.path.exists(output_file_basename + '.bngl'):
        print "WARNING! File %s already exists!" % (output_file_basename + '.bngl')
        output_file_basename += '_1'

    bng_filename = output_file_basename + '.bngl'
    gdat_filename = output_file_basename + '.gdat'
    cdat_filename = output_file_basename + '.cdat'
    net_filename = output_file_basename + '.net'

    output = StringIO() 
    bng_file = open(bng_filename, 'w')
    bng_file.write(gen.get_content())
    bng_file.write(run_ssa_code)
    bng_file.close()
    p = subprocess.Popen(['perl', BNG._get_bng_path(), bng_filename],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (p_out, p_err) = p.communicate()
def gather_data(model,output_dir):
    trajectories = []
    files = os.listdir(output_dir )
    cleanup=True
    for filename in files:
        if filename.endswith('.cdat'):
            data = None
            
            head,tail = filename.split(".")
            cdat = numpy.loadtxt(output_dir + '/' +head+'.cdat',skiprows=1)
            print np.shape(cdat),' cdat shape'
            if len(model.observables):
                gdat = numpy.loadtxt(output_dir + '/' +head+'.gdat',skiprows=1)[:,1:] # exclude first column (time)
                
            else:
                gdat = numpy.ndarray((len(cdat_arr), 0))
                
            names = ['time'] + ['__s%d' % i for i in range(cdat.shape[1]-1)] # -1 for time column
            yfull_dtype = zip(names, itertools.repeat(float))
            if len(model.observables):
                names += model.observables.keys()
            data = np.column_stack((cdat,gdat))
            print np.shape(data),' data shape'
            data = data.view(dtype=[(n, 'float64') for n in names])
            print np.shape(data),' data shape after'
            if cleanup ==True:
                os.unlink(output_dir + '/' +head+'.cdat')
                os.unlink(output_dir + '/' +head+'.gdat')
                os.unlink(output_dir + '/' +head+'.net')
                os.unlink(output_dir + '/' +head+'.bngl')
            trajectories.append(data)
    return trajectories

