"""
Wrapper functions for running the Kappa programs Kasim and complx.

In general only the following three functions will be needed:
* :py:func:`run_simulation`
* :py:func:`influence_map`
* :py:func:`contact_map`

The other functions are used internally and manage the execution of the Kappa
software and the parsing of the data into a Numpy format.
"""

__author__ = "johnbachman"

import pysb
from pysb.generator.kappa import KappaGenerator
import os
import subprocess
import random
import re
import sympy
import numpy as np

def run_simulation(model, **kwargs):
  """Runs the model using kasim with the specified arguments for time and
  number of points (note that it also generates the influence and flux
  maps, though they are not used here).

  Returns the kasim simulation data as a numpy array, which can be
  plotted using the plot command.
  """

  outs = run_kasim(model, **kwargs)
  return _parse_kasim_outfile(outs['out'])

def influence_map(model, do_open=False, **kwargs):
  """Runs Kasim with no simulation events, which generates the influence map.

  If do_open is set to True, then calls the open_file method to display
  the influence map using the default program for opening .gv files
  (e.g., GraphViz).
  """

  kasim_dict = run_kasim(model, time=0, points=0, **kwargs)
  im_filename = kasim_dict['im']

  if do_open:
    open_file(im_filename)


def contact_map(model, do_open=False, **kwargs):
  """Runs complx with the appropriate arguments for generating the contact map.

  If do_open is True, attempts to open the JPG file for display.
  """

  gen = KappaGenerator(model, dialect='complx')
  #kappa_filename = '%d_%d_temp.ka' % (os.getpid(), random.randint(0, 10000))
  kappa_filename = '%s.ka' % model.name
  jpg_filename = kappa_filename.replace('.ka', '.jpg')
  dot_filename = kappa_filename.replace('.ka', '.dot')
  args = ['--output-high-res-contact-map-jpg', jpg_filename,
          '--output-high-res-contact-map-dot', dot_filename]
  run_complx(gen, kappa_filename, args, **kwargs)

  if do_open:
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
      raise Exception(p.stderr.read())

  except Exception as e:
    raise Exception("problem running complx: " + str(e))


def run_kasim(model, time=10000, points=200, output_dir='.', cleanup=False):
  """Run kasim with the provided arguments.
  """

  gen = KappaGenerator(model)
  #kappa_filename = '%d_%d_temp.ka' % (os.getpid(), random.randint(0, 10000))

  kappa_filename = '%s/%s_%d_%d_temp.ka' % (output_dir,
                        model.name, os.getpid(), random.randint(0, 10000))

  im_filename = kappa_filename.replace('.ka', '_im.gv')
  fm_filename = kappa_filename.replace('.ka', '_fm.gv')
  out_filename = kappa_filename.replace('.ka', '.out')

  args = ['-i', kappa_filename, '-t', str(time), '-p', str(points),
          '-o', out_filename, '-im', im_filename, '-flux', fm_filename]

  try:
    kappa_file = open(kappa_filename, 'w')
    kappa_file.write(gen.get_content())
    kappa_file.close()

    print "Running kasim"
    p = subprocess.Popen(['KaSim'] + args)
                            #stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.communicate()

    if p.returncode:
      raise Exception(p.stdout.read())

  except Exception as e:
    raise Exception("problem running KaSim: " + str(e))

  finally:
    if cleanup:
      for filename in [kappa_filename, im_filename,
                      fm_filename, out_filename]:
        if os.access(filename, os.F_OK):
          os.unlink(filename)

  output_dict = {'out':out_filename, 'im':im_filename, 'fm':'flux.dot'}
  return output_dict


def _parse_kasim_outfile(out_filename):
  """Parse the outputfile produced by kasim."""

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


def open_file(filename):
  """Utility function for opening files for display on Mac OS X
  (jpg, gv, dot, etc.). Ultimately this should be rewritten to auto-detect
  the operating system and use the appropriate system call.
  """

  try:
    p = subprocess.Popen(['open'] + [filename],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #p.communicate()
    p.wait()
    if p.returncode:
      raise Exception(p.stderr.read())
  except Exception as e:
    raise Exception("Problem opening file: ", e)


def generate_influence_map(model):
  """DEPRECATED, since Kasim always generates the flux and influence maps
  when run."""

  gen = KappaGenerator(model)
  kappa_filename = '%d_%d_temp.ka' % (os.getpid(), random.randint(0, 10000))
  dot_filename = kappa_filename.replace('.ka', '.jpg')
  args = ['--output-influence-map-jpg', jpg_filename]
  run_complx(gen, kappa_filename, args)



