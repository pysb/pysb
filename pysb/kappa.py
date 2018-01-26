"""
Wrapper functions for running the Kappa programs *KaSim* and *KaSa*.

The path to the directory containing the KaSim and KaSa executables can be
specified in one of three ways:

- set the KAPPAPATH environment variable to the KaSim directory
- move Kappa to /usr/local/share/KaSim (macOS, Linux) or
  c:\Program Files\KaSim (Windows)
- set the path using the :py:func:`pysb.pathfinder.set_path` function at
  runtime
"""

from __future__ import print_function as _
import pysb
import pysb.pathfinder as pf
from pysb.generator.kappa import KappaGenerator
import os
import subprocess
import re
import numpy as np
import tempfile
import shutil
import warnings
from collections import namedtuple
from pysb.util import read_dot

try:
    from future_builtins import zip
except ImportError:
    pass


def set_kappa_path(path):
    """Set the path to the KaSim and KaSa executables.

    Deprecated. Use pysb.pathfinder.set_path() instead.

    Parameters
    ----------
    path: string
        Directory containing KaSim and KaSa executables.
    """
    warnings.warn("Function %s() is deprecated; use "
                  "pysb.pathfinder.set_path() instead" %
                  set_kappa_path.__name__, category=DeprecationWarning,
                  stacklevel=2)
    pf.set_path('kasim', path)
    pf.set_path('kasa', path)


class KasimInterfaceError(RuntimeError):
    pass


class KasaInterfaceError(RuntimeError):
    pass

StaticAnalysisResult = namedtuple('StaticAnalysisResult',
                                  ['contact_map', 'influence_map'])

SimulationResult = namedtuple('SimulationResult',
                                  ['timecourse', 'flux_map'])


def run_simulation(model, time=10000, points=200, cleanup=True,
                   output_prefix=None, output_dir=None, flux_map=False,
                   perturbation=None, seed=None, verbose=False):
    """Runs the given model using KaSim and returns the parsed results.

    Parameters
    ----------
    model : pysb.core.Model
        The model to simulate/analyze using KaSim.
    time : number
        The amount of time (in arbitrary units) to run a simulation.
        Identical to the -u time -l argument when using KaSim at the command
        line.
        Default value is 10000. If set to 0, no simulation will be run.
    points : integer
        The number of data points to collect for plotting.
        Note that this is not identical to the -p argument of KaSim when
        called from the command line, which denotes plot period (time interval
        between points in plot).
        Default value is 200. Note that the number of points actually returned
        by the simulator will be points + 1 (including the 0 point).
    cleanup : boolean
        Specifies whether output files produced by KaSim should be deleted
        after execution is completed. Default value is True.
    output_prefix: str
        Prefix of the temporary directory name. Default is
        'tmpKappa_<model name>_'.
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
    seed : integer
        A seed integer for KaSim random number generator. Set to None to
        allow KaSim to use a random seed (default) or supply a seed for
        deterministic behaviour (e.g. for testing)
    verbose : boolean
        Whether to pass the output of KaSim through to stdout/stderr.

    Returns
    -------
    If flux_map is False, returns the kasim simulation data as a Numpy ndarray.
    Data is accessed using the syntax::

            results[index_name]

    The index 'time' gives the time coordinates of the simulation. Data for the
    observables can be accessed by indexing the array with the names of the
    observables. Each entry in the ndarray has length points + 1, due to the
    inclusion of both the zero point and the final timepoint.

    If flux_map is True, returns an instance of SimulationResult, a namedtuple
    with two members, `timecourse` and `flux_map`. The `timecourse` field
    contains the simulation ndarray, and the `flux_map` field is an instance of
    a networkx MultiGraph containing the flux map. For details on viewing
    the flux map graphically see :func:`run_static_analysis` (notes section).
    """

    gen = KappaGenerator(model)

    if output_prefix is None:
        output_prefix = 'tmpKappa_%s_' % model.name

    base_directory = tempfile.mkdtemp(prefix=output_prefix, dir=output_dir)

    base_filename = os.path.join(base_directory, model.name)
    kappa_filename = base_filename + '.ka'
    fm_filename = base_filename + '_fm.dot'
    out_filename = base_filename + '.out'

    if points == 0:
        raise ValueError('The number of data points cannot be zero.')
    plot_period = (float(time) / points) if time > 0 else 1.0

    args = ['-i', kappa_filename, '-u', 'time', '-l', str(time),
            '-p', '%.5f' % plot_period, '-o', out_filename]

    if seed:
        args.extend(['-seed', str(seed)])

    # Generate the Kappa model code from the PySB model and write it to
    # the Kappa file:
    with open(kappa_filename, 'w') as kappa_file:
        kappa_file.write(gen.get_content())
        # If desired, add instructions to the kappa file to generate the
        # flux map:
        if flux_map:
            kappa_file.write('%%mod: [true] do $DIN "%s" [true];\n' %
                             fm_filename)

        # If any perturbation language code has been passed in, add it to
        # the Kappa file:
        if perturbation:
            kappa_file.write('\n%s\n' % perturbation)

    # Run KaSim
    kasim_path = pf.get_path('kasim')
    p = subprocess.Popen([kasim_path] + args,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=base_directory)
    if verbose:
        for line in iter(p.stdout.readline, b''):
            print('@@', line, end='')
    (p_out, p_err) = p.communicate()

    if p.returncode:
        raise KasimInterfaceError(
            p_out.decode('utf8') + '\n' + p_err.decode('utf8'))

    # The simulation data, as a numpy array
    data = _parse_kasim_outfile(out_filename)

    if flux_map:
        try:
            flux_graph = read_dot(fm_filename)
        except ImportError:
            if cleanup:
                raise
            else:
                warnings.warn(
                        "The pydot library could not be "
                        "imported, so no MultiGraph "
                        "object returned (returning None); flux map "
                        "dot file available at %s" % fm_filename)
                flux_graph = None

    if cleanup:
        shutil.rmtree(base_directory)

    # If a flux map was generated, return both the simulation output and the
    # flux map as a networkx multigraph
    if flux_map:
        return SimulationResult(data, flux_graph)
    # If no flux map was requested, return only the simulation data
    else:
        return data


def run_static_analysis(model, influence_map=False, contact_map=False,
                        cleanup=True, output_prefix=None, output_dir=None,
                        verbose=False):
    """Run static analysis (KaSa) on to get the contact and influence maps.

    If neither influence_map nor contact_map are set to True, then a ValueError
    is raised.

    Parameters
    ----------
    model : pysb.core.Model
        The model to simulate/analyze using KaSa.
    influence_map : boolean
        Whether to compute the influence map.
    contact_map : boolean
        Whether to compute the contact map.
    cleanup : boolean
        Specifies whether output files produced by KaSa should be deleted
        after execution is completed. Default value is True.
    output_prefix: str
        Prefix of the temporary directory name. Default is
        'tmpKappa_<model name>_'.
    output_dir : string
        The directory in which to create the temporary directory for
        the .ka and other output files. Defaults to the system temporary file
        directory (e.g. /tmp). If the specified directory does not exist,
        an Exception is thrown.
    verbose : boolean
        Whether to pass the output of KaSa through to stdout/stderr.

    Returns
    -------
    StaticAnalysisResult, a namedtuple with two fields, `contact_map` and
    `influence_map`, each containing the respective result as an instance
    of a networkx MultiGraph. If the either the contact_map or influence_map
    argument to the function is False, the corresponding entry in the
    StaticAnalysisResult returned by the function will be None.

    Notes
    -----
    To view a networkx file graphically, use `draw_network`::

        import networkx as nx
        nx.draw_networkx(g, with_labels=True)

    You can use `graphviz_layout` to use graphviz for layout (requires pydot
    library)::

        import networkx as nx
        pos = nx.drawing.nx_pydot.graphviz_layout(g, prog='dot')
        nx.draw_networkx(g, pos, with_labels=True)

    For further information, see the networkx documentation on visualization:
    https://networkx.github.io/documentation/latest/reference/drawing.html
    """

    # Make sure the user has asked for an output!
    if not influence_map and not contact_map:
        raise ValueError('Either contact_map or influence_map (or both) must '
                         'be set to True in order to perform static analysis.')

    gen = KappaGenerator(model, _warn_no_ic=False)

    if output_prefix is None:
        output_prefix = 'tmpKappa_%s_' % model.name

    base_directory = tempfile.mkdtemp(prefix=output_prefix, dir=output_dir)

    base_filename = os.path.join(base_directory, str(model.name))
    kappa_filename = base_filename + '.ka'
    im_filename = base_filename + '_im.dot'
    cm_filename = base_filename + '_cm.dot'

    # NOTE: in the args passed to KaSa, the directory for the .dot files is
    # specified by the --output_directory option, and the output_contact_map
    # and output_influence_map should only be the base filenames (without
    # a directory prefix).
    # Contact map args:
    if contact_map:
        cm_args = ['--compute-contact-map', '--output-contact-map',
                   os.path.basename(cm_filename),
                   '--output-contact-map-directory', base_directory]
    else:
        cm_args = ['--no-compute-contact-map']
    # Influence map args:
    if influence_map:
        im_args = ['--compute-influence-map', '--output-influence-map',
                   os.path.basename(im_filename),
                   '--output-influence-map-directory', base_directory]
    else:
        im_args = ['--no-compute-influence-map']
    # Full arg list
    args = [kappa_filename] + cm_args + im_args

    # Generate the Kappa model code from the PySB model and write it to
    # the Kappa file:
    with open(kappa_filename, 'w') as kappa_file:
        kappa_file.write(gen.get_content())

    # Run KaSa using the given args
    kasa_path = pf.get_path('kasa')
    p = subprocess.Popen([kasa_path] + args,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=base_directory)
    if verbose:
        for line in iter(p.stdout.readline, b''):
            print('@@', line, end='')
    (p_out, p_err) = p.communicate()

    if p.returncode:
        raise KasaInterfaceError(
            p_out.decode('utf8') + '\n' + p_err.decode('utf8'))

    # Try to create the graphviz objects from the .dot files created
    try:
        # Convert the contact map to a Graph
        cmap = read_dot(cm_filename) if contact_map else None
        imap = read_dot(im_filename) if influence_map else None
    except ImportError:
        if cleanup:
            raise
        else:
            warnings.warn(
                    "The pydot library could not be "
                    "imported, so no MultiGraph "
                    "object returned (returning None); "
                    "contact/influence maps available at %s" %
                    base_directory)
            cmap = None
            imap = None

    # Clean up the temp directory if desired
    if cleanup:
        shutil.rmtree(base_directory)

    return StaticAnalysisResult(cmap, imap)


def contact_map(model, **kwargs):
    """Generates the contact map via KaSa.

    Parameters
    ----------
    model : pysb.core.Model
        The model for generating the influence map.
    **kwargs : other keyword arguments
        Any other keyword arguments are passed to the function
        :py:func:`run_static_analysis`.

    Returns
    -------
    networkx MultiGraph object containing the contact map. For details on
    viewing the contact map graphically see :func:`run_static_analysis` (notes
    section).
    """
    kasa_result = run_static_analysis(model, influence_map=False,
                                    contact_map=True, **kwargs)
    return kasa_result.contact_map


def influence_map(model, **kwargs):
    """Generates the influence map via KaSa.

    Parameters
    ----------
    model : pysb.core.Model
        The model for generating the influence map.
    **kwargs : other keyword arguments
        Any other keyword arguments are passed to the function
        :py:func:`run_static_analysis`.

    Returns
    -------
    networkx MultiGraph object containing the influence map. For details on
    viewing the influence map graphically see :func:`run_static_analysis`
    (notes section).
    """

    kasa_result = run_static_analysis(model, influence_map=True,
                                    contact_map=False, **kwargs)
    return kasa_result.influence_map

### "PRIVATE" Functions ###############################################

def _parse_kasim_outfile(out_filename):
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
        with open(out_filename, 'r') as fh:
            for header_line in fh:
                if header_line[0] != '#':
                    break

            # Load the output file as a numpy record array, skip the name row
            arr = np.loadtxt(fh, dtype=float, delimiter=',')

        raw_names = [term.strip() for term in re.split(',', header_line)]
        column_names = []

        # Get rid of the quotes surrounding the observable names
        for raw_name in raw_names:
            mo = re.match('"(.*)"', raw_name)
            if mo:
                name = mo.group(1)
                # Rename the time column to remain backwards compatible
                if name == '[T]':
                    name = 'time'
                column_names.append(name)
            else:
                column_names.append(raw_name)

        # Create the dtype argument for the numpy record array
        dt = list(zip(column_names, ('float', ) * len(column_names)))

        recarr = arr.view(dt)
    except Exception as e:
        raise Exception("problem parsing KaSim outfile: " + str(e))

    return recarr
