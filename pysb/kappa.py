"""
Wrapper functions for running the Kappa programs *Kasim* and *complx*.

In general only the following three functions will be needed for typical use:

    * :py:func:`run_simulation`
    * :py:func:`influence_map`
    * :py:func:`contact_map`

The other functions are used internally and manage the execution of the Kappa
software and the parsing of the data into a Numpy format.
"""

from __future__ import print_function as _
import pysb
from pysb.generator.kappa import KappaGenerator
import os
import subprocess
import random
import re
import sympy
import numpy as np
import tempfile
import shutil
import warnings

try:
    from future_builtins import zip
except ImportError:
    pass

class KasimInterfaceError(RuntimeError):
    pass

def run_simulation(model, time=10000, points=200, cleanup=True,
                   output_prefix=None, output_dir=None, flux_map=False,
                   perturbation=None, verbose=False):
    """Runs the given model using KaSim and returns the parsed results.

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
        Default value is 200. Note that the number of points returned by the
        simulator will be points + 1 (including the 0 point).
    cleanup : boolean
        Specifies whether output files produced by KaSim should be deleted
        after execution is completed. Default value is False.
    output_prefix: str
        Prefix of the temporary directory name. Default is 'tmpKappa'.
    output_dir : string
        The directory in which to create the temporary directory for
        the .ka and other output files. Defaults to the system temporary file
        directory (e.g. /tmp). If the specified directory does not exist,
        an Exception is thrown.
    flux_map: boolean
        Specifies whether or not to produce the flux map (generated over the
        full duration of the simulation). Default value is False.
    perturbation : string or None
        Optional perturbation language syntax to be appended to the Kappa file.
        See KaSim manual for more details. Default value is None (no
        perturbation).

    Returns
    -------
    If flux_map is False, returns the kasim simulation data as a Numpy ndarray.
    Data is accessed using the syntax::

            results[index_name]

    The index 'time' gives the data for the time coordinates of the
    simulation. Data for the observables can be accessed by indexing the
    array with the names of the observables. Each entry in the ndarray
    has length points + 1, due to the inclusion of both the 0 point and the
    final timepoint.

    If flux_map is True, returns a two-tuple whose first element is the
    simulation ndarray, and whose second element is an instance of a pygraphviz
    AGraph containing the flux map. The flux map can be rendered as a pdf
    using the dot layout program as follows::

        fluxmap.draw('fluxmap.pdf', prog='dot')
    """

    gen = KappaGenerator(model)

    if output_prefix is None:
        output_prefix = 'tmpKappa_%s_' % model.name

    base_directory = tempfile.mkdtemp(prefix=output_prefix, dir=output_dir)

    base_filename = os.path.join(base_directory, model.name)
    kappa_filename = base_filename + '.ka'
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
        # flux map:
        if flux_map:
            kappa_file.write('%%mod: [true] do $FLUX "%s" [true]\n' %
                             fm_filename)

        # If any perturbation language code has been passed in, add it to the
        # Kappa file:
        if perturbation:
            kappa_file.write('\n%s\n' % perturbation)

        kappa_file.close()

        p = subprocess.Popen(['kasim'] + args,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if verbose:
            for line in iter(p.stdout.readline, b''):
                print('@@', line, end='')
        (p_out, p_err) = p.communicate()

        if p.returncode:
            raise KasimInterfaceError(p_out + '\n' + p_err)

        # The simulation data, as a numpy array
        data = parse_kasim_outfile(out_filename)

        if flux_map:
            try:
                import pygraphviz
                flux_graph = pygraphviz.AGraph(fm_filename)
            except ImportError:
                if cleanup:
                    raise ImportError(
                            "Couldn't import pygraphviz, which is "
                            "required to return the flux map as a "
                            "pygraphviz AGraph object. Either install "
                            "pygraphviz or set cleanup=False to retain "
                            "dot files.")
                else:
                    warnings.warn(
                            "pygraphviz could not be imported so no AGraph "
                            "object returned (returning None); flux map "
                            "dot file available at %s" % fm_filename)
                    flux_graph = None

    except Exception as e:
        raise Exception("Problem running KaSim: " + str(e))

    finally:
        if cleanup:
            shutil.rmtree(base_directory)

    # If a flux map was generated, return both the simulation output and the
    # flux map as a pygraphviz graph
    if flux_map:
        return (data, flux_graph)
    # If no flux map was requested, return only the simulation data
    else:
        return data


def influence_map(model, do_open=False, **kwargs):
    """Generates the influence map via KaSa.

    Parameters
    ----------
    model : pysb.core.Model
        The model for generating the influence map.
    do_open : boolean
        If do_open is set to True, then calls the :py:func:`open_file` method
        to display the influence map using the default program for opening .dot
        files (e.g., GraphViz).
    **kwargs : other keyword arguments
        Any other keyword arguments are passed to the function
        :py:func:`run_kasa`.

    Returns
    -------
    string
        Returns the name of the .dot file where the influence map
        has been stored.
    """

    kasa_dict = run_kasa(model, influence_map=True, contact_map=False,
                         **kwargs)
    im_filename = kasa_dict['im']

    if do_open:
        open_file(im_filename)

    return im_filename

def contact_map(model, do_open=False, **kwargs):
    """Generates the contact map via KaSa.

    Parameters
    ----------
    model : pysb.core.Model
        The model for generating the influence map.
    do_open : boolean
        If do_open is set to True, then calls the :py:func:`open_file` method
        to display the influence map using the default program for opening .dot
        files (e.g., GraphViz).
    **kwargs : other keyword arguments
        Any other keyword arguments are passed to the function
        :py:func:`run_kasa`.

    Returns
    -------
    string
        Returns the name of the .dot file where the contact map
        has been stored.
    """

    kasa_dict = run_kasa(model, influence_map=False, contact_map=True,
                         **kwargs)
    cm_filename = kasa_dict['cm']

    if do_open:
        open_file(cm_filename)

    return cm_filename


### "PRIVATE" Functions ###############################################

def run_complx(gen, kappa_filename, args):
    """Generalized method for passing arguments to the complx executable.

    *DEPRECATED* because complx itself is deprecated. Switching over to using
    KaSa for static analysis.

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

    warnings.warn("Complx is no longer supported, please use run_kasa instead",
                  DeprecationWarning, stacklevel=2)

    try:
        kappa_file = open(kappa_filename, 'w')
        kappa_file.write(gen.get_content())
        kappa_file.close()
        cmd = 'complx ' + ' '.join(args) + ' ' + kappa_filename
        print("Command: " + cmd)
        p = subprocess.Popen(['complx'] + args + [kappa_filename],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #p.communicate()
        p.wait()

        if p.returncode:
            raise Exception(p.stderr.read())

    except Exception as e:
        raise Exception("problem running complx: " + str(e))



def run_kasa(model, influence_map=False, contact_map=False, output_dir='.',
             cleanup=False, base_filename=None):
    """Run KaSa (static analyzer) on the given model.

    Parameters
    ----------
    model : pysb.core.Model
        The model to simulate/analyze using KaSa.
    influence_map : boolean
        Whether to compute the influence map.
    contact_map : boolean
        Whether to compute the contact map.
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
        contact map filename appends '_cm.dot'.

    Returns
    -------
    A dict with two entries giving the filenames for the files produced

        * output_dict['im'] gives the influence map filename, or None if not
          produced
        * output_dict['cm'] gives the contact map filename, or None if not
          produced
    """

    gen = KappaGenerator(model)

    if not base_filename:
        base_filename = '%s/%s_%d_%d_temp' % (output_dir,
                        model.name, os.getpid(), random.randint(0, 10000))

    kappa_filename = base_filename + '.ka'
    im_filename = base_filename + '_im.dot'
    cm_filename = base_filename + '_cm.dot'

    # Contact map args
    if contact_map:
        cm_args = ['--compute-contact-map', '--output-contact-map',
                   cm_filename]
    else:
        cm_args = ['--no-compute-contact-map']
    # Influence map args
    if influence_map:
        im_args = ['--compute-influence-map', '--output-influence-map',
                   im_filename]
    else:
        im_args = ['--no-compute-influence-map']
    # Full arg list
    args = [kappa_filename, '--output-directory', output_dir] \
            + cm_args + im_args

    try:
        kappa_file = open(kappa_filename, 'w')

        # Generate the Kappa model code from the PySB model and write it to
        # the Kappa file:
        kappa_file.write(gen.get_content())
        kappa_file.close()

        print("Running KaSa")
        p = subprocess.Popen(['KaSa'] + args)
                           #stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.communicate()

        if p.returncode:
            raise Exception(p.stdout.read())

    except Exception as e:
        raise Exception("Problem running KaSa: " + str(e))

    finally:
        if cleanup:
            for filename in [kappa_filename, im_filename, cm_filename]:
                if os.access(filename, os.F_OK):
                    os.unlink(filename)

    output_dict = {'im':im_filename, 'cm':cm_filename}
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
        dt = list(zip(column_names, ('float',)*len(column_names)))

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
