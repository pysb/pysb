"""
Wrapper functions for running the Kappa programs *Kasim* and *complx*.

In general only the following three functions will be needed for typical use:

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
    """Runs the given model using KaSim and returns the parsed results.

    Parameters
    ----------
    **kwargs : List of keyword arguments
        All keyword arguments specifying conditions for the simulation are
        passed to the function :py:func:`run_kasim` (see documentation
        associated with that function for more information).

    Returns
    -------
    numpy.ndarray
        Returns the kasim simulation data as a Numpy ndarray. Data is accessed
        using the syntax::
        
            results[index_name]
            
        The index 'time' gives the data for the time coordinates of the
        simulation. Data for the observables can be accessed by indexing the
        array with the names of the observables.
    """

    outs = run_kasim(model, **kwargs)
    return parse_kasim_outfile(outs['out'])

def influence_map(model, do_open=False, **kwargs):
    """Generates the influence map.

    Runs KaSim with no events or points and sets the dump_influence_map
    argument to True.

    Parameters
    ----------
    model : pysb.core.Model
        The model for generating the influence map.
    do_open : boolean
        If do_open is set to True, then calls the :py:func:`open_file` method
        to display the influence map using the default program for opening .gv
        files (e.g., GraphViz).
    **kwargs : other keyword arguments
        Any other keyword arguments are passed to the function
        :py:func:`run_kasim`.

    Returns
    -------
    string
        Returns the name of the .gv (GraphViz) file where the influence map
        has been stored. This can subsequently be used to build a networkx or
        PyGraphViz graph.
    """

    kasim_dict = run_kasim(model, time=0, points=0, dump_influence_map=True,
                           **kwargs)
    im_filename = kasim_dict['im']

    if do_open:
        open_file(im_filename)

    return im_filename

def contact_map(model, output_dir='.', base_filename=None, do_open=False, 
                **kwargs):
    """Runs complx with arguments for generating the contact map.

    Parameters
    ----------
    model : pysb.core.Model
        The model for generating the contact map.
    output_dir : string
        The subdirectory in which to generate the Kappa (.ka) file for the
        model and all output files produced by complx. Default value is '.'
        Note that only relative paths can be specified; paths are relative
        to the directory where the current Python instance is running.
        If the specified directory does not exist, an Exception is thrown.
    base_filename : The base filename to be used for generation of the Kappa
        (.ka) file and all output files produced by complx. Defaults to a
        string of the form::

            '%s_%d_%d_temp' % (model.name, os.getpid(), random.randint(0,10000))

        The contact map filenames append '_cm.jpg' and '_cm.dot' to this base
        filename; the reachable complexes filename appends '_rch.dot'.
    do_open : boolean
        If do_open is set to True, then calls the :py:func:`open_file` method
        to display the contact map using the default program for opening .jpg
        files.
    **kwargs : other keyword arguments
        Any other keyword arguments are passed through to the function
        :py:func:`run_complx`.

    """

    gen = KappaGenerator(model, dialect='complx')

    if not base_filename:
        base_filename = '%s/%s_%d_%d_temp' % (output_dir,
                         model.name, os.getpid(), random.randint(0, 10000))

    kappa_filename = base_filename + '.ka'
    jpg_filename = base_filename + '_cm.jpg'
    dot_filename = base_filename + '_cm.dot'
    reachables_filename = base_filename + '_rch.dot'

    args = ['--output-high-res-contact-map-jpg', jpg_filename,
            '--output-high-res-contact-map-dot', dot_filename,
            '--output-reachable-complexes', reachables_filename]
    run_complx(gen, kappa_filename, args, **kwargs)

    if do_open:
        open_file(jpg_filename)


### "PRIVATE" Functions ###############################################

def run_complx(gen, kappa_filename, args):
    """Generalized method for passing arguments to the complx executable.
   
    Parameters
    ----------
    gen : :py:class:`pysb.generator.KappaGenerator`
        A KappaGenerator object that is used to produce the Kappa content
        for writing to a file.
    kappa_filename : string
        The name of the file to write the generated Kappa to.
    args : list of strings
        List of command line arguments to pass to complx, with one entry for
        each argument, for example::

            ['--output-high-res-contact-map-jpg', jpg_filename]
    """

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


def run_kasim(model, time=10000, points=200, output_dir='.', cleanup=False,
              base_filename=None, dump_influence_map=False,
              perturbation=None):
    """Run KaSim on the given model with the provided arguments.

    Parameters
    ----------
    model : pysb.core.Model
        The model to simulate/analyze using KaSim.
    time : number
        The amount of time (in arbitrary units) to run a simulation.
        Identical to the -t argument when using KaSim at the command line.
        Default value is 10000. If set to 0, no simulation will be run, but
        the influence map will be generated (if dump_influence_map is set to
        True).
    points : integer
        The number of data points to collect for plotting.
        Identical to the -p argument when using KaSim at the command line.
        Default value is 200.
    output_dir : string
        The subdirectory in which to generate the Kappa (.ka) file for the
        model and all output files produced by KaSim. Default value is '.'
        Note that only relative paths can be specified; paths are relative
        to the directory where the current Python instance is running.
        If the specified directory does not exist, an Exception is thrown.
    cleanup : boolean
        Specifies whether output files produced by KaSim should be deleted
        after execution is completed. Default value is False.
    base_filename : The base filename to be used for generation of the Kappa
        (.ka) file and all output files produced by KaSim. Defaults to a
        string of the form::

            '%s_%d_%d_temp' % (model.name, program id, random.randint(0,10000))

        The influence map filename appends '_im.dot' to this base filename; the
        flux map filename appends '_fm.dot'; and the simulation output file
        appends '.out'
    dump_influence_map : boolean
        Specifies whether or not to produce the influence map. Default value
        is False.
    perturbation : string or None
        Optional perturbation language syntax to be appended to the Kappa file.
        See KaSim manual for more details. Default value is None (no
        perturbation).

    Returns
    -------
    A dict with three entries giving the filenames for the files produced:

        * output_dict['out'] gives the .out filename
        * output_dict['im'] gives the influence map filename
        * output_dict['fm'] gives the flux map filename
    """

    gen = KappaGenerator(model)

    if not base_filename:
        base_filename = '%s/%s_%d_%d_temp' % (output_dir,
                        model.name, os.getpid(), random.randint(0, 10000))

    kappa_filename = base_filename + '.ka'
    im_filename = base_filename + '_im.dot'
    fm_filename = base_filename + '_fm.dot'
    out_filename = base_filename + '.out'

    args = ['-i', kappa_filename, '-t', str(time), '-p', str(points),
            '-o', out_filename]

    try:
        kappa_file = open(kappa_filename, 'w')

        # Generate the Kappa model code from the PySB model and write it to
        # the Kappa file:
        kappa_file.write(gen.get_content())

        # If desired, add instructions to the kappa file to generate the
        # influence map:
        if dump_influence_map:
            kappa_file.write('%def: "dumpInfluenceMap" "true"\n')
            kappa_file.write('%%def: "influenceMapFileName" "%s"\n\n' %
                             im_filename)

        # If any perturbation language code has been passed in, add it to the
        # Kappa file:
        if perturbation:
            kappa_file.write('\n%s\n' % perturbation)

        kappa_file.close()

        print "Running kasim"
        p = subprocess.Popen(['KaSim'] + args)
                           #stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.communicate()

        if p.returncode:
            raise Exception(p.stdout.read())

    except Exception as e:
        raise Exception("Problem running KaSim: " + str(e))

    finally:
        if cleanup:
            for filename in [kappa_filename, im_filename,
                            fm_filename, out_filename]:
                if os.access(filename, os.F_OK):
                    os.unlink(filename)

    output_dict = {'out':out_filename, 'im':im_filename, 'fm':'flux.dot'}
    return output_dict


def parse_kasim_outfile(out_filename):
    """
    Parses the KaSim .out file into a Numpy ndarray.

    Parameters
    ----------
    out_filename : string
        String specifying the location of the .out filename produced by KaSim.

    Returns
    -------
    numpy.ndarray
        Returns the KaSim simulation data as a Numpy ndarray. Data is accessed
        using the syntax::
        
            results[index_name]
            
        The index 'time' gives the data for the time coordinates of the
        simulation. Data for the observables can be accessed by indexing the
        array with the names of the observables.
    """

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
    """Utility function for opening files for display on Mac OS X.

    Uses the 'open' command to open the given file using the default program
    associated with the file's filetype. Ultimately this should be rewritten to
    auto-detect the operating system and use the appropriate system call.
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




