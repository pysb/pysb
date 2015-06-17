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
import pysb.bng  
from pysb.simulate import Simulator
from pysb.bng import generate_equations
import numpy as np
import pysb.generator.bng
import multiprocessing
class BNGSSASimulator(Simulator):
        
    def __init__(self, model, tspan=None, cleanup=True, verbose=False):
        super(BNGSSASimulator, self).__init__(model, tspan, verbose)
        generate_equations(self.model, cleanup, self.verbose)
        generate_network(self.model, cleanup, self.verbose)
    
    def run(self, tspan=None, param_values=None,initial_changes=None,  output_dir=os.getcwd(),output_file_basename=None,cleanup=False, n_runs=1,**additional_args):

        if tspan is not None:
            self.tspan = tspan
        elif self.tspan is None:
            raise Exception("'tspan' must be defined.")
        
        trajectories = bng_simulate(self.model, tspan=self.tspan, param_values=param_values, initial_changes=initial_changes,\
                                    output_dir=output_dir,output_file_basename=output_file_basename,verbose=self.verbose,\
                                    n_runs=n_runs, **additional_args)
    
        self.tout = np.array(trajectories)[:,:,0] + self.tspan[0]
        # species
        self.y = np.array(trajectories)[:,:,1:]
        # observables and expressions
        self._calc_yobs_yexpr(param_values)
    
    def _calc_yobs_yexpr(self, param_values=None):
        super(BNGSSASimulator, self)._calc_yobs_yexpr()
        
    def get_yfull(self):
        return super(BNGSSASimulator, self).get_yfull()

def run_ssa(model, tspan, param_values=None,initial_changes=None,  output_dir=os.getcwd(), output_file_basename=None,\
             cleanup=True, verbose=False, n_runs=1,**additional_args):

    sim = BNGSSASimulator(model,tspan, verbose=verbose)
    sim.run( tspan, param_values, initial_changes,  output_dir, output_file_basename, cleanup, n_runs,**additional_args)
    yfull = sim.get_yfull()
    return sim.tout, yfull


def bng_simulate(model,tspan, param_values=None,initial_changes=None, output_dir=os.getcwd(), output_file_basename=None, \
                 cleanup=True, verbose=False, n_runs=1,**additional_args):

    
    
    if output_file_basename is None:
        process_id = multiprocessing.current_process()
        process_id = str(process_id.pid)
        output_file_basename = '%s_%s_runfile' % (model.name,process_id)
    output_file_basename1 = '%s_runfile' % model.name
    
    bng_run = output_file_basename1.rstrip('_runfile')+'_temp.net'
    
    if not os.path.exists(bng_run ):
        print "WARNING! File %s does not exist!" % (output_file_basename + '.net')
        
    bng_filename = output_file_basename + '.bngl'
    gdat_filename = output_file_basename + '.gdat'
    cdat_filename = output_file_basename + '.cdat'
    net_filename = output_file_basename + '.net'
    
    if param_values is not None:
        if len(param_values) != len(model.parameters):
            raise Exception("param_values must be the same length as model.parameters")
        for i in range(len(param_values)):
            model.parameters[i].value = param_values[i]

    if initial_changes is not None:
        original_values = {}
        for cp, value_obj in model.initial_conditions:
                    if value_obj.name in initial_changes:
                        original_values[value_obj.name] = value_obj.value
                        value_obj.value = initial_changes[value_obj.name]
                        
    ssa_args = "t_start=>%s,sample_times=>%s" % (str(tspan[0]),str(list(tspan)))
    for key,val in additional_args.items(): ssa_args += ", %s=>%s" % (key,"\""+str(val)+"\"" if isinstance(val,str) else str(val))
    if verbose: ssa_args += ", verbose=>1"

    run_ssa_code = "readFile({prefix=>\"%s\",file=>\"%s\"})\n" %(output_file_basename,bng_run)
    
    gen = generate_parameters(model)
    run_ssa_code += gen
    if n_runs == 1: 
        run_ssa_code += "simulate({method=>\"ssa\",%s, %s })\n" % (ssa_args, "prefix=>\""+output_file_basename+str(n)+"\"")
    else:
        for n in range(n_runs):
            run_ssa_code += "simulate({method=>\"ssa\",%s, %s })\n" % (ssa_args, "prefix=>\""+output_file_basename+str(n)+"\"")
            run_ssa_code += "resetConcentrations()\n"
    
    output = StringIO()
    working_dir = os.getcwd()
    os.chdir(output_dir)
    bng_file = open(bng_filename, 'w')
    bng_file.write(run_ssa_code)
    bng_file.close()
    p = subprocess.Popen(['perl', pysb.bng._get_bng_path(), bng_filename],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if verbose:
        for line in iter(p.stdout.readline, b''):
            print line,
    (p_out, p_err) = p.communicate()
    if p.returncode:
        raise pysb.bng.GenerateNetworkError(p_out.rstrip("at line")+"\n"+p_err.rstrip())
    if initial_changes is not None:
        for cp, value_obj in model.initial_conditions:
            if value_obj.name in original_values:
                value_obj.value = original_values[value_obj.name]
    
    trajectories = []
    for i in xrange(n_runs):
        trajectories.append(numpy.loadtxt(os.path.join(output_dir, output_file_basename+str(i)+".cdat")))

    # Clean up
    if cleanup:
        for i in xrange(n_runs):
            if os.path.exists(os.path.join(output_dir, output_file_basename+str(i)+'.net')):
                os.unlink(os.path.join(output_dir,output_file_basename+ str(i)+'.net'))
            if os.path.exists(os.path.join(output_dir, output_file_basename+str(i)+'.cdat')):
                os.unlink(os.path.join(output_dir,output_file_basename+ str(i)+'.cdat'))
            if os.path.exists(os.path.join(output_dir,output_file_basename+ str(i)+'.gdat')):
                os.unlink(os.path.join(output_dir,output_file_basename+ str(i)+'.gdat'))
    # Move to working directory  
    os.chdir(working_dir)
    return trajectories

def generate_network(model, cleanup=True, append_stdout=False, verbose=False):
    gen = BngGenerator(model)
    if not model.initial_conditions:
        raise NoInitialConditionsError()
    if not model.rules:
        raise NoRulesError()
    bng_filename = '%s_temp.bngl' % (model.name)
    net_filename = bng_filename.replace('.bngl', '.net')
    output = StringIO()
    try:
        bng_file = open(bng_filename, 'w')
        bng_file.write(gen.get_content())
        bng_file.write(pysb.bng._generate_network_code)
        bng_file.close()
        p = subprocess.Popen(['perl', pysb.bng._get_bng_path(), bng_filename],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if verbose:
            for line in iter(p.stdout.readline, b''):
                print line,
        (p_out, p_err) = p.communicate()
        if p.returncode:
            raise pysb.bng.GenerateNetworkError(p_out.rstrip()+"\n"+p_err.rstrip())
        net_file = open(net_filename, 'r')
        output.write(net_file.read())
        net_file.close()
        if append_stdout:
            output.write("#\n# BioNetGen execution log follows\n# ==========")
            output.write(re.sub(r'(^|\n)', r'\n# ', p_out))
    finally:
        print 'BNGL net file created'
    return output.getvalue()

def generate_parameters(model):
    content = ''
    exprs = model.expressions_constant()
    max_length = max(len(p.name) for p in
                     model.parameters | model.expressions)
    for p in model.parameters:
        #print (("\tsetParameter(\"%s\",   %e)\n") %(p.name, p.value))
        content += (("setParameter(\"%s\",   %e)\n") %(p.name, p.value))
    for e in exprs:
        #print  (("\tsetParameter(\"%s\",   %s\n)") % (e.name, pysb.generator.bngsympy_to_muparser(e.expr)))
        content += (("setParameter(\"%s\",   %s\n)") % (e.name, pysb.generator.bng.sympy_to_muparser(e.expr)))
    content += "\n"
    return content