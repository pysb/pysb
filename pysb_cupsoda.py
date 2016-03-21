# coding=utf-8
import os
import re
import subprocess
import time
import warnings
import tempfile
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    use_pycuda = True
except ImportError, e:
    use_pycuda = False
    pass
import numpy as np
import pandas
from scipy.constants import N_A
from pysb.bng import generate_equations
from pysb.simulate import Simulator

read_csv = pandas.read_csv

_cupsoda_path = None

# Putting this here because the cupSODA class may be generalized in the future
# to a MultiSolver. All supported integrators should be included here,
# even if they don't have default options. Obviously, the only integrator
# currently supported is "cupSODA" (case insensitive).
default_integrator_options = {
    'cupsoda': {
        'max_steps': 20000,  # max no. of internal iterations (LSODA's MXSTEP)
        'atol': 1e-8,  # absolute tolerance
        'rtol': 1e-8,  # relative tolerance
        'n_blocks': None,  # number of GPU blocks
        'gpu': 0,  # which GPU
        'vol': None,  # volume
        'obs_species_only': True,  # print only observable species
        'load_ydata': True,  # load conc data after simulation
        'memory_usage': 'global'  # global memory (see _memory_options dict)
    },
}


def set_cupsoda_path(directory):
    global _cupsoda_path
    _cupsoda_path = os.path.join(directory, 'cupSODA')
    # Make sure file exists and that it is not a directory
    if not os.access(_cupsoda_path, os.F_OK) or not \
            os.path.isfile(_cupsoda_path):
        raise Exception('Could not find cupSODA binary in ' +
                        os.path.abspath(directory) + '.')
    # Make sure file has executable permissions
    elif not os.access(_cupsoda_path, os.X_OK):
        raise Exception("cupSODA binary in " + os.path.abspath(directory) +
                        " does not have executable permissions.")


def _get_cupsoda_path():
    """
    Return the path to the cupSODA executable.

    Looks for the cupSODA executable in a user-defined location set via
    ``set_cupsoda_path``, the environment variable CUPSODAPATH or in a few
    hard-coded standard locations.
    """

    global _cupsoda_path

    # Just return cached value if it's available
    if _cupsoda_path:
        return _cupsoda_path

    path_var = 'CUPSODAPATH'
    bin_dirs = [
        '/usr/local/share/',
        'c:/Program Files/',
    ]

    def check_bin_dir(bin_dir):
        # Return the full path to the cupSODA executable or False if it
        # can't be found in one of the expected places.
        bin_path = os.path.join(bin_dir, 'cupSODA')
        if os.access(bin_path, os.F_OK):
            return bin_path
        else:
            return False

    # First check the environment variable, which has the highest precedence
    if path_var in os.environ:
        script_path = check_bin_dir(os.environ[path_var])
        if not script_path:
            raise Exception('Environment variable %s is set but the path could'
                            ' not be found, is not accessible or does not '
                            'contain a cupSODA executable file.' % path_var)
    # If the environment variable isn't set, check the standard locations
    # Check the standard locations for the executable
    else:
        for b in bin_dirs:
            bin_path = check_bin_dir(b)
            if bin_path:
                break
            else:
                raise Exception('Could not find cupSODA installed in one of '
                                'the following locations:' +
                                ''.join('\n    ' + x for x in bin_dirs) +
                                '\nPlease put the executable (or a link to '
                                'it) in one of these locations or set the '
                                'path using set_cupsoda_path().')

    # Cache path for future use
    _cupsoda_path = bin_path
    return bin_path


class CupsodaSolver(Simulator):
    """An interface for running cupSODA, a CUDA implementation of LSODA.
    Parameters
    ----------
    model : pysb.Model
        Model to integrate.
    tspan : vector-like, optional (default: None)
        Time values at which the integrations are sampled. The first and last
        values define the time range.
    cleanup : bool, optional
        If True (default), delete the temporary files after the simulation is
        finished. If False, leave them in place. Useful for debugging.
    verbose : bool, optional (default: False)
        Verbose output 
    integrator : string, optional (default: 'cupsoda')
        Name of the integrator to use, taken from the integrators listed in 
        pysb.tools.cupSODA.default_integrator_options.
    integrator_options :
        * max_steps        : max no. of internal iterations (LSODA's MXSTEP)
                             (default: None)
        * atol             : absolute tolerance (default: 1e-8)
        * rtol             : relative tolerance (default: 1e-8)
        * n_blocks         : number of GPU blocks (default: 64)
        * gpu              : index of GPU to run on (default: 0)
        * vol              : volume (required if number units; default: None)
        * obs_species_only : print only the concentrations of species in
                             observables (default: True)
        * load_ydata       : read species concentrations from output files
                             after simulation (default: True)
        * memory_usage     : type of memory usage (default: 'global')
                             * 'global': global memory (suggested for
                               medium-large models)
                             * 'shared': shared memory
                             * 'sharedconstant': both shared and constant
                               memory

    Attributes
    ----------
    verbose: bool
        Verbose output.
    model : pysb.Model
        Model passed to the constructor.
    tspan : vector-like, optional 
        Time values passed to the constructor.
    tout: numpy.ndarray
        Time points returned by the simulator (may be different from ``tspan``
        if simulation is interrupted for some reason).
    y : numpy.ndarray
        Species trajectories for each simulation. Dimensionality is
        ``(n_sims, len(tspan), len(model.species))``.
    yobs : numpy.ndarray (with record-style data-type)
        Observable trajectories for each simulation. Dimensionality is
        ``(n_sims, len(tspan))``; record names follow ``model.observables`` 
        names.
    yobs_view : numpy.ndarray
        Array view (sharing the same data buffer) on ``yobs``. 
        Dimensionality is ``(n_sims, len(tspan), len(model.observables))``.
    yexpr : numpy.ndarray with record-style data-type
        Expression trajectories. Dimensionality is ``(n_sims, len(tspan))`` 
        and record names follow ``model.expressions_dynamic()`` names.
    yexpr_view : numpy.ndarray
        An array view (sharing the same data buffer) on ``yexpr``.
        Dimensionality is ``(n_sims, len(tspan),
        len(model.expressions_dynamic()))``.
    outdir : string, optional (default: os.getcwd())
            Output directory.

    Notes
    -----
    1) The expensive step of generating the reaction network is performed
       during initialization only if the network does not already exist
       (len(model.reactions)==0 and len(model.species)==0) OR if it is
       explicitly requested (gen_net==True).
    2) The cupSODA class is derived from the Solver class in pysb.integrate
       although it overrides all of the functions of the Solver class (i.e,
       __init__() and run()). This was done with the idea that eventually a
       lower level BaseSolver class will be created from which the Solver
       and cupSODA classes will be derived. This will probably be necessary
       for performing multiscale simulations where one would want to embed a
       (potentially different) solver within each entity (e.g., cell) in the
       simulation environment. Note that this is also why the argument list
       for the __init__() function is identical to that of the Solver class.
    3) Like the Solver class, the cupSODA class accepts an optional
       'integrator' string argument. However, if anything other than
       'cupsoda' (case insensitive) is passed an Exception is thrown. The
       idea behind this design is that the cupSODA class may eventually be
       generalized to a MultiSolver class, for which cupSODA is just one of
       multiple supported integrators. This is analogous to the Solver
       class, which currently supports all of the integrators included in
       :py:class:`scipy.integrate.ode`.
    4) The arrays 'y', 'yobs', and 'yobs_view' from the
       pysb.integrate.Solver base class have been overridden as dictionaries in
       which the keys are simulation numbers (ints) and the values are the
       same ndarrays as in the base class. For efficiency, these objects are
       initialized as empty dictionaries and only filled if the
       'load_conc_data' argument to cupSODA.run() is set to True (the
       default) or if the cupSODA.load_data() method is called directly.
    """

    _memory_options = {'global': '0', 'shared': '1', 'sharedconstant': '2'}

    def __init__(self, model, tspan=None, cleanup=True, verbose=False,
                 integrator='cupsoda', **integrator_options):
        super(CupsodaSolver, self).__init__(model, tspan, verbose)
        generate_equations(self.model, cleanup, self.verbose)

        self.model_parameter_rules = self.model.parameters_rules()
        self.len_rxns = len(self.model.reactions)
        self.len_species = len(self.model.species)
        self.model_rxns = self.model.reactions
        self.tspan_length = len(self.tspan)
        self.outdir = None
        # Set integrator options to defaults
        self.options = {}
        integrator = integrator.lower()  # case insensitive
        if default_integrator_options.get(integrator.lower()):
            self.options.update(default_integrator_options[integrator])
        else:
            raise Exception(
                "Integrator type '" + integrator + "' not recognized.")

        # overwrite default integrator options
        self.options.update(integrator_options)

    def run(self, param_values, y0, tspan=None, outdir=None,
            prefix=None, **integrator_options):
        """Perform a set of integrations.

        Returns nothing; if load_ydata=True, can access the object's 
        ``y``, ``yobs``, or ``yobs_view`` attributes to retrieve the results.

        Parameters
        ----------
        param_values : list-like
            Rate constants for all reactions for all simulations. Dimensions
            are ``(N_SIMS, len(model.reactions))``.
        y0 : list-like
            Initial species concentrations for all reactions for all
            simulations. Dimensions are ``(N_SIMS, len(model.species))``.
        tspan : vector-like, optional
            Time values (exception thrown if set to None both here and in
            constructor).
        outdir : string, optional (default: None)
            Output directory. Note that a directory is created within the
            specified one using ``tempfile.mkdtemp``. If None, a system
            temporary directory is used. The location is accessible via the
            ``outdir`` attribute.
        prefix : string, optional (default: model.name)
            Output files will be named "prefix_i", for i=[0,N_SIMS). The
            prefix is also used for the output directory (see above argument).
        integrator_options : See CupsodaSolver constructor.
        
        Notes
        -----
        If 'vol' is provided, cupSODA will assume that species counts are in
        number units  and will automatically convert them to concentrations
        by dividing by N_A*vol (N_A = Avogadro's number).
        
        If load_ydata=True and obs_species_only=True, concentrations of
        species not in observables are set to 'nan'.
        """

        start_time = time.time()
        # make sure tspan is defined somewhere
        if tspan is not None:
            self.tspan = tspan
        elif self.tspan is None:
            raise Exception("'tspan' must be defined.")

        # overwrite default integrator options
        self.options.update(integrator_options)

        # just to be safe
        param_values = np.array(param_values)
        y0 = np.array(y0)

        # Error checks on 'param_values' and 'y0'
        if len(param_values) != len(y0):
            raise Exception("Lengths of 'param_values' (len=" + str(
                len(param_values)) + ") and 'y0' (len=" + str(
                len(y0)) + ") must be equal.")
        if len(param_values.shape) != 2 or param_values.shape[1] != len(
                self.model.parameters):
            raise Exception(
                "'param_values' must be a 2D array with dimensions N_SIMS x "
                "len(model.parameters): param_values.shape=" +
                str(param_values.shape))
        if len(y0.shape) != 2 or y0.shape[1] != len(self.model.species):
            raise Exception(
                "'y0' must be a 2D array with dimensions N_SIMS x "
                "len(model.species): y0.shape=" + str(y0.shape))

        # Default prefix is model name
        if not prefix:
            prefix = self.model.name

        # Create outdir if it doesn't exist
        self.outdir = tempfile.mkdtemp(prefix=prefix, dir=outdir)
        if self.verbose:
            print("Output directory is %s" % self.outdir)

        # Path to cupSODA executable
        bin_path = _get_cupsoda_path()

        # Put the cupSODA input files within the output directory
        cupsoda_files = os.path.join(self.outdir, "__CUPSODA_FILES")
        os.mkdir(cupsoda_files)

        # Simple default for number of blocks
        n_blocks = self.options.get('n_blocks')
        gpu = self.options.get('gpu')
        if not gpu:
            gpu = 0
        if not n_blocks:
            default_threads_per_block = 16
            n_species = len(self.model.species)
            bytes_per_float = 4
            # +1 for time variable
            memory_per_thread = (n_species + 1) * bytes_per_float
            n_sims = len(param_values)
            if use_pycuda:
                device = cuda.Device(gpu)
                attrs = device.get_attributes()
                shared_memory_per_block = attrs[
                  pycuda._driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]
                upper_limit_threads_per_block = attrs[
                  pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK]
                max_threads_per_block = min(
                  shared_memory_per_block / memory_per_thread,
                  upper_limit_threads_per_block)
                threads_per_block = min(max_threads_per_block,
                                        default_threads_per_block)
                n_blocks = int(np.ceil(1. * n_sims / threads_per_block))
                if self.verbose:
                    print('Shared_mem_per_block/mem_per_block = %f' % (shared_memory_per_block / memory_per_thread))
                    print('Threads per block %i ' % threads_per_block)
                    print('Number of blocks %i ' % n_blocks)
            else:
                n_blocks = int(
                    np.ceil(1. * n_sims) / default_threads_per_block)

        # Create c_matrix
        c_matrix = np.zeros((len(param_values), self.len_rxns))
        par_names = [p.name for p in self.model_parameter_rules]
        rate_params = self.model_parameter_rules
        rate_mask = np.array([p in rate_params for p in self.model.parameters])
        par_dict = {par_names[i]: i for i in range(len(par_names))}
        rate_args = []
        param_values = param_values[:, rate_mask]
        rate_order = []
        for rxn in self.model_rxns:
            rate_args.append([arg for arg in rxn['rate'].args if
                              not re.match("_*s", str(arg))])
            reactants = 0
            for i in rxn['reactants']:
                if not str(self.model.species[i]) == '__source()':
                    reactants += 1
            rate_order.append(reactants)
        output = 0.01 * len(param_values)
        output = int(output) if output > 1 else 1
        for i in range(len(param_values)):
            if self.verbose and i % output == 0:
                print(str(int(round(100. * i / len(param_values)))) + "%")
            for j in range(self.len_rxns):
                if self.options['vol']:
                    rate = 1 * (N_A * self.options['vol']) ** \
                               (rate_order[j] - 1)
                else:
                    rate = 1.0
                for r in rate_args[j]:
                    x = str(r)
                    if x in par_names:
                        rate *= param_values[i][par_dict[x]]
                    else:
                        # FIXME: need to detect non-numbers and throw an error
                        rate *= float(x)
                c_matrix[i][j] = rate

        if self.verbose:
            print("100%")

        # Create cupSODA input files
        self._create_input_files(cupsoda_files, c_matrix, y0)
        end_time = time.time()
        if self.verbose:
            print("Set up time = %4.4f " % (end_time - start_time))
        # Build command
        # ./cupSODA input_model_folder blocks output_folder simulation_
        # file_prefix gpu_number fitness_calculation memory_use dump
        command = [bin_path, cupsoda_files, str(n_blocks),
                   os.path.join(self.outdir, "Output"), prefix,
                   str(self.options['gpu']),
                   '0', self._memory_usage,
                   '1' if self.verbose else '0']
        print("Running cupSODA: " + ' '.join(command))

        # Run simulation
        start_time = time.time()
        p = subprocess.Popen(command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)

        cupsoda_time = -1
        for line in iter(p.stdout.readline, b''):
            if line.startswith('Running'):
                cupsoda_time = float(
                    line.split(':')[1].replace('seconds', ''))
            print(">>> " + line.rstrip())
        # subprocess.call(command)
        end_time = time.time()
        total_time = end_time - start_time
        if self.verbose:
            print("cupSODA = %4.4f" % cupsoda_time)
            print("Total time = %4.4f " % (end_time - start_time))
            print("Total - cupSODA = %4.4f" % (total_time - cupsoda_time))

        # Load concentration data if requested   
        if self.options.get('load_ydata'):
            start_time = time.time()
            self._get_y(prefix=prefix)
            end_time = time.time()
            if self.verbose:
                print("Get_y time = %4.4f " % (end_time - start_time))
            start_time = time.time()
            self._calc_yobs_yexpr(param_values)
            end_time = time.time()
            if self.verbose:
                print("Calc yopbs_yepr time = %4.4f " % (end_time - start_time))

    @property
    def _memory_usage(self):
        try:
            return self._memory_options[self.options['memory_usage']]
        except KeyError:
            raise Exception('memory_usage must be one of %s',
                            self._memory_options.keys())

    def _create_input_files(self, cupsoda_files, param_values, y0):

        # Number of sims, rxns, and species
        n_sims, n_rxns = param_values.shape
        n_species = len(y0[0])

        # atol_vector 
        with open(os.path.join(cupsoda_files, "atol_vector"),
                  'wb') as atol_vector:
            for i in range(self.len_species):
                atol_vector.write(str(self.options.get('atol')))
                if i < self.len_species - 1:
                    atol_vector.write("\n")

        # c_matrix
        with open(os.path.join(cupsoda_files, "c_matrix"), 'wb') as c_matrix:
            for i in range(n_sims):
                line = ""
                for j in range(n_rxns):
                    if j > 0:
                        line += "\t"
                    line += str(param_values[i][j])
                c_matrix.write(line)
                if i < n_sims - 1:
                    c_matrix.write("\n")

        # cs_vector
        with open(os.path.join(cupsoda_files, "cs_vector"), 'wb') as cs_vector:
            out_species = range(self.len_species)
            if self.options.get('obs_species_only'):
                out_species = [False for sp in self.model.species]
                for obs in self.model.observables:
                    for i in obs.species:
                        out_species[i] = True
                out_species = [i for i in range(len(out_species)) if
                               out_species[i]]
            for i in range(len(out_species)):
                if i > 0:
                    cs_vector.write("\n")
                cs_vector.write(str(out_species[i]))

        # left_side
        with open(os.path.join(cupsoda_files, "left_side"), 'wb') as left_side:
            for i in range(self.len_rxns):
                line = ""
                for j in range(self.len_species):
                    if j > 0:
                        line += "\t"
                    stoich = 0
                    for k in self.model_rxns[i]['reactants']:
                        if j == k:
                            stoich += 1
                    line += str(stoich)
                if i < self.len_rxns - 1:
                    left_side.write(line + "\n")
                else:
                    left_side.write(line)

        # max_steps
        with open(os.path.join(cupsoda_files, "max_steps"), 'wb') as mxsteps:
            mxsteps.write(str(self.options['max_steps']))

        # model_kind
        with open(os.path.join(cupsoda_files, "modelkind"),
                  'wb') as model_kind:
            model_kind.write("deterministic")
            vol = self.options['vol']
            # volume
            if vol:
                # Set population of __source() to N_A*vol and warn the user
                warnings.warn("Number units detected in cupSODA.run(). Setting the population \
                              of __source() (if it exists) equal to %g*%g." % (
                    N_A, vol))
                y0 = y0 / (N_A * vol)
                for i, sp in enumerate(self.model.species):
                    if str(sp) == '__source()':
                        y0[:, i] = 1.
                        break

        # MX_0
        with open(os.path.join(cupsoda_files, "MX_0"), 'wb') as MX_0:
            for i in range(n_sims):
                line = ""
                for j in range(n_species):
                    if j > 0:
                        line += "\t"
                    line += str(y0[i][j])
                MX_0.write(line)
                if i < n_sims - 1:
                    MX_0.write("\n")

        # M_feed
        with open(os.path.join(cupsoda_files, "M_feed"), 'wb') as M_feed:
            line = ""
            for j, sp in enumerate(self.model.species):
                if j > 0:
                    line += "\t"
                if str(sp) == '__source()':
                    line += '1'
                else:
                    line += '0'
            M_feed.write(line)

        # right_side
        with open(os.path.join(cupsoda_files, "right_side"),
                  'wb') as right_side:
            for i in range(self.len_rxns):
                line = ""
                for j in range(self.len_species):
                    if j > 0:
                        line += "\t"
                    stochiometry = 0
                    for k in self.model_rxns[i]['products']:
                        if j == k:
                            stochiometry += 1
                    line += str(stochiometry)
                if i < self.len_rxns - 1:
                    right_side.write(line + "\n")
                else:
                    right_side.write(line)

        # rtol
        with open(os.path.join(cupsoda_files, "rtol"), 'wb') as rtol:
            rtol.write(str(self.options.get('rtol')))

        # t_vector
        with open(os.path.join(cupsoda_files, "t_vector"), 'wb') as t_vector:
            t_vector.write("\n")
            for t in self.tspan:
                t_vector.write(str(float(t)) + "\n")

        # time_max
        with open(os.path.join(cupsoda_files, "time_max"), 'wb') as time_max:
            time_max.write(str(float(self.tspan[-1])))

    def _get_y(self, indir=None, prefix=None, out_species=None):
        """Read simulation results from output files. 

        Returns nothing. Fills the ``y`` and ``tout`` arrays.

        Parameters
        ----------
        indir : string, optional (default: self.outdir)
            Directory where data files are located.
        prefix : string, optional (default: model.name)
            Prefix for output filenames (e.g., "egfr" for egfr_0,...).
        out_species: integer or list of integers, optional 
            Indices of species present in the data file (e.g., in
            cupSODA.run(), if obs_species_only=True then only species
            involved in observables are output to file). If not specified,
            an attempt will be made to read them from the  'cs_vector' file
            in 'indir/__CUPSODA_FILES/'.
        """

        if indir is None:
            indir = os.path.join(self.outdir, "Output")
        if prefix is None:
            prefix = self.model.name
        if out_species is None:
            try:
                out_species = np.loadtxt(
                    os.path.join(self.outdir, "__CUPSODA_FILES", "cs_vector"),
                    dtype=int)
            except IOError:
                print("ERROR: Cannot determine which species have been printed to file. Either provide an")
                print("'out_species' array or place the '__CUPSODA_FILES' directory in the input directory.")
                raise

        files = [filename for filename in os.listdir(indir) if
                 re.match(prefix, filename)]
        if len(files) == 0:
            raise Exception("Cannot find any output files to load data from.")

        self.y = len(files) * [None]
        self.tout = len(files) * [None]

        option = 1
        y_n = np.ones((self.tspan_length, self.len_species)) * float('nan')
        t_out = np.ones(self.tspan_length) * float('nan')
        for n in range(len(self.y)):
            self.y[n] = y_n.copy()
            self.tout[n] = t_out.copy()
            # Read data from file
            filename = os.path.join(indir, prefix) + "_" + str(n)
            if not os.path.isfile(filename):
                raise Exception("Cannot find input file " + filename)
            # if self.verbose:
            #    print "Reading " + filename + " ...",
            # Optimizing the time to read in data back to Python
            # When reading in data for the first simulation, it checks to see if reading the data line
            # line is faster, or if using pandas.read_csv is faster.
            # For the remaining data that is read in, it uses the faster of the two methods.
            if n == 0:
                start_time = time.time()
            if option == 1:
                with open(filename, 'rb') as f:
                    for i in range(len(self.y[n])):
                        data = f.readline().split()
                        self.tout[n][i] = data[0]
                        # exclude first column (time)
                        self.y[n][i, out_species] = data[1:]
                if self.options['vol']:
                    self.y[n][:, :] *= self.options['vol'] * N_A
            if n == 0:
                end_time = time.time()
                method_1 = end_time - start_time
                start_time = time.time()
            if n == 0 or option == 2:
                data = read_csv(filename, sep='\t', skiprows=None, header=None)
                data = data.as_matrix()
                self.tout[n] = data[:, 0]
                if self.options['vol']:
                    self.y[n][:, out_species] = data[:, 1:] * \
                                                self.options['vol'] * N_A
                else:
                    self.y[n][:, out_species] = data[:, 1:]
            if n == 0:
                end_time = time.time()
                method_2 = end_time - start_time
                if method_1 > method_2:
                    option = 2
        self.tout = np.array(self.tout)
        self.y = np.asarray(self.y)

    def _calc_yobs_yexpr(self, param_values=None):
        super(CupsodaSolver, self)._calc_yobs_yexpr()

    def get_yfull(self):
        return super(CupsodaSolver, self).get_yfull()


def run_cupsoda(model, tspan, param_values, y0, outdir=os.getcwd(),
                prefix=None, verbose=False, **integrator_options):
    sim = CupsodaSolver(model, verbose=verbose, **integrator_options)
    sim.run(param_values, y0, tspan, outdir, prefix)
    if sim.options.get('load_ydata'):
        yfull = sim.get_yfull()
        return sim.tout, yfull
