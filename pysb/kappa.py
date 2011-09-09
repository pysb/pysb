import pysb
from pysb.generator.kappa import KappaGenerator
import os
import subprocess
import random
import re
import sympy
import numpy as np
import pygraphviz as pgv

# not ideal, but it will work for now during development

def run_complx(gen, kappa_filename, args):
  try:
      kappa_file = open(kappa_filename, 'w')
      kappa_file.write(gen.get_content())
      kappa_file.close()
      cmd = 'complx ' + ' '.join(args) + ' ' + kappa_filename
      print "Command: " + cmd
      p = subprocess.Popen(['complx'] + args + [kappa_filename],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      #p.communicate()
      p.wait()

      if p.returncode:
          #raise Exception(p.stdout.read())
          raise Exception(p.stderr.read())

      #contact_map = pgv.AGraph(dot_filename)
      #contact_map.layout()
      #contact_map.draw(kappa_filename.replace('.ka', '.png'))
  except Exception as e:
      raise Exception("problem running complx: " + str(e))
  #finally:
      #for filename in [kappa_filename, dot_filename]:
      #    if os.access(filename, os.F_OK):
      #        os.unlink(filename)
# end generate_contact_map

def generate_influence_map(model):
  gen = KappaGenerator(model)
  kappa_filename = '%d_%d_temp.ka' % (os.getpid(), random.randint(0, 10000))
  jpg_filename = kappa_filename.replace('.ka', '.jpg')
  args = ['--output-influence-map-jpg', jpg_filename]
  run_complx(gen, kappa_filename, args)

def generate_contact_map(model):
  gen = KappaGenerator(model)
  kappa_filename = '%d_%d_temp.ka' % (os.getpid(), random.randint(0, 10000))
  jpg_filename = kappa_filename.replace('.ka', '.jpg')
  args = ['--output-high-res-contact-map-jpg', jpg_filename]
  run_complx(gen, kappa_filename, args)

def run_kasim(model, time=10000, num_datapts=200):
  gen = KappaGenerator(model)
  kappa_filename = '%d_%d_temp.ka' % (os.getpid(), random.randint(0, 10000))
  im_filename = kappa_filename.replace('.ka', '_im.gv')
  fm_filename = kappa_filename.replace('.ka', '_fm.gv')
  out_filename = kappa_filename.replace('.ka', '.out')

  args = ['-i', kappa_filename, '-t', str(time), '-p', str(num_datapts),
          '-o', out_filename, '-im', im_filename, '-flux', fm_filename]

  try:
      kappa_file = open(kappa_filename, 'w')
      kappa_file.write(gen.get_content())
      kappa_file.close()
      p = subprocess.Popen(['KaSim'] + args)
                            #stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      p.communicate()

      if p.returncode:
          raise Exception(p.stdout.read())

  except Exception as e:
      raise Exception("problem running KaSim: " + str(e))
  #finally:
      #for filename in [kappa_filename, dot_filename]:
      #    if os.access(filename, os.F_OK):
      #        os.unlink(filename)

  output_dict = {'out':out_filename, 'im':im_filename, 'fm':'flux.dot'}
  return output_dict

# end run_kasim

def get_kasim_data(model, time=10000, num_datapts=1000):
  outs = run_kasim(model, time, num_datapts)
  return parse_kasim_outfile(outs['out'])

def parse_kasim_outfile(out_filename):
  try:
    out_file = open(out_filename, 'r')
    line = out_file.readline().strip()
    out_file.close()

    line = line[2:]  # strip off opening '# '
    raw_names = re.split(' ', line)
    column_names = []

    # Get ride of the quotes surrounding the observable names
    for raw_name in raw_names:
      mo = re.match("'(.*)'", raw_name)
      if (mo): column_names.append(mo.group(1))
      else: column_names.append(raw_name)

    dt = zip(column_names, ('float',)*len(column_names))

    # Load as numpy record array, skip the name row
    arr = np.loadtxt(out_filename, dtype=dt, skiprows=1)
  except Exception as e:
    raise Exception("problem parsing KaSim outfile: " + str(e))

  return arr
