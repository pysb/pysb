import pysb
from pysb.generator.kappa import KappaGenerator
import os
import subprocess
import random
import re
import sympy
import numpy as np
import pygraphviz as pgv

### "PUBLIC" Functions ################################################
# i.e., functions that are most useful to be called externally


# Runs the model using kasim with the specified arguments for time and
# number of points (note that it also generates the influence and flux
# maps, though they are not used here).
#
# Returns the kasim simulation data as a numpy array, which can be
# plotted using the plot command.
def get_kasim_data(model, time=10000, points=1000):
  outs = run_kasim(model, time, points)
  return parse_kasim_outfile(outs['out'])


# Runs Kasim with no simulation events, which generates the influence map,
# and then displays it using GraphViz (assumes that GraphViz is set up and is the
# default program for opening .gv files)
def show_influence_map(model):
  kasim_dict = run_kasim(model, time=0, points=0)
  im_filename = kasim_dict['im']
  open_file(im_filename)


# Runs complx with the appropriate arguments for generating the contact map.
# DOESN'T WORK: WHY???
def show_contact_map(model):
  gen = KappaGenerator(model, dialect='complx')
  #kappa_filename = '%d_%d_temp.ka' % (os.getpid(), random.randint(0, 10000))
  kappa_filename = '%s.ka' % model.name
  jpg_filename = kappa_filename.replace('.ka', '.jpg')
  args = ['--output-high-res-contact-map-jpg', jpg_filename]
  run_complx(gen, kappa_filename, args)
  open_file(jpg_filename)


### "PRIVATE" Functions ###############################################

# Generalized method for passing arguments to the complx executable.
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





# Runs kasim, which
def run_kasim(model, time=10000, points=200):
  gen = KappaGenerator(model)
  #kappa_filename = '%d_%d_temp.ka' % (os.getpid(), random.randint(0, 10000))
  kappa_filename = '%s.ka' % model.name
  im_filename = kappa_filename.replace('.ka', '_im.gv')
  fm_filename = kappa_filename.replace('.ka', '_fm.gv')
  out_filename = kappa_filename.replace('.ka', '.out')

  args = ['-i', kappa_filename, '-t', str(time), '-p', str(points),
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


# Parses the outputfile produced by kasim, which has the form
def parse_kasim_outfile(out_filename):
  try:
    out_file = open(out_filename, 'r')

    line = out_file.readline().strip() # Get the first line
    out_file.close()
    line = line[2:]  # strip off opening '# '
    raw_names = re.split(' ', line)
    column_names = []

    # Get rid of the quotes surrounding the observable names
    for raw_name in raw_names:
      mo = re.match("'(.*)'", raw_name)
      if (mo): column_names.append(mo.group(1))
      else: column_names.append(raw_name)

    # Create the dtype argument for the numpy record array
    dt = zip(column_names, ('float',)*len(column_names))

    # Load the output file as a numpy record array, skip the name row
    arr = np.loadtxt(out_filename, dtype=dt, skiprows=1)

  except Exception as e:
    raise Exception("problem parsing KaSim outfile: " + str(e))

  return arr


# Utility function for opening files for display (jpg, gv, dot, etc.)
def open_file(filename):
  try:
      p = subprocess.Popen(['open'] + [filename],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      #p.communicate()
      p.wait()
      if p.returncode:
          raise Exception(p.stderr.read())
  except Exception as e:
      raise Exception("Problem opening file: ", e)


# DEPRECATED! Since Kasim always generates the flux and influence maps when run
def generate_influence_map(model):
  gen = KappaGenerator(model)
  kappa_filename = '%d_%d_temp.ka' % (os.getpid(), random.randint(0, 10000))
  dot_filename = kappa_filename.replace('.ka', '.jpg')
  args = ['--output-influence-map-jpg', jpg_filename]
  run_complx(gen, kappa_filename, args)



