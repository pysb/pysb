from __future__ import print_function as _
from pysb.simulator.base import Simulator, SimulatorException, SimulationResult
import pysb
import pysb.bng
import numpy as np
from scipy.constants import N_A
import os
import re
import subprocess
import tempfile
import time
import logging
from pysb.logging import EXTENDED_DEBUG
import shutil
from pysb.pathfinder import get_path
import sympy

try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import pycuda.driver as cuda
except ImportError:
    cuda = None


class CupSodaSimulator(Simulator):
    """An interface for running cupSODA, a CUDA implementation of LSODA.

    cupSODA is a graphics processing unit (GPU)-based implementation of the
    LSODA simulation algorithm (see references). It requires an NVIDIA GPU
    card with support for the CUDA framework version 7 or above. Further
    details of cupSODA and software can be found on github:
    https://github.com/aresio/cupSODA

    The simplest way to install cupSODA is to use a pre-compiled version,
    which can be downloaded from here:
    https://github.com/aresio/cupSODA/releases

    Parameters
    ----------

    model : pysb.Model
        Model to integrate.
    tspan : vector-like, optional
        Time values at which the integrations are sampled. The first and last
        values define the time range.
    initials : list-like, optional
        Initial species concentrations for all simulations. Dimensions are 
        N_SIMS x number of species.
    param_values : list-like, optional
        Parameters for all simulations. Dimensions are N_SIMS x number of 
        parameters.
    verbose : bool or int, optional
        Verbosity level, see :class:`pysb.simulator.base.Simulator` for
        further details.
    **kwargs: dict, optional
        Extra keyword arguments, including:

        * ``gpu``: Index of GPU to run on (default: 0)
        * ``vol``: System volume; required if model encoded in extrinsic 
          (number) units (default: None)
        * ``obs_species_only``: Only output species contained in observables
          (default: True) 
        * ``cleanup``: Delete all temporary files after the simulation is 
          finished. Includes both BioNetGen and cupSODA files. Useful for 
          debugging (default: True)
        * ``prefix``: Prefix for the temporary directory containing cupSODA
          input and output files (default: model name)
        * ``base_dir``: Directory in which temporary directory with cupSODA
          input and output files are placed (default: system directory
          determined by `tempfile.mkdtemp`)
        * ``integrator``: Name of the integrator to use; see 
          `default_integrator_options` (default: 'cupsoda')
        * ``integrator_options``: A dictionary of keyword arguments to
          supply to the integrator; see `default_integrator_options`.

    Attributes
    ----------

    model : pysb.Model
        Model passed to the constructor.
    tspan : numpy.ndarray
        Time values passed to the constructor.
    initials : numpy.ndarray
        Initial species concentrations for all simulations. Dimensions are 
        number of simulations x number of species.
    param_values : numpy.ndarray
        Parameters for all simulations. Dimensions are number of simulations 
        x number of parameters.
    verbose: bool or int
        Verbosity setting. See the base class
        :class:`pysb.simulator.base.Simulator` for further details.
    gpu : int
        Index of GPU being run on
    vol : float or None
        System volume
    n_blocks: int
        Number of GPU blocks used by the simulator.
    outdir : str
        Directory where cupSODA output files are placed. Input files are
        also placed here.
    opts: dict
        Dictionary of options for the integrator in use.
    integrator : str
        Name of the integrator in use.
    default_integrator_options : dict
        Nested dictionary of default options for all supported integrators.

    Notes
    -----

    1. If `vol` is defined, species amounts and rate constants are assumed
       to be in number units and are automatically converted to concentration
       units before generating the cupSODA input files. The species
       concentrations returned by cupSODA are converted back to number units
       during loading.

    2. If `obs_species_only` is True, only the species contained within 
       observables are output by cupSODA. All other concentrations are set 
       to 'nan'.

    References
    ----------

    1. Nobile M.S., Cazzaniga P., Besozzi D., Mauri G., 2014. GPU-accelerated
       simulations of mass-action kinetics models with cupSODA, Journal of
       Supercomputing, 69(1), pp.17-24.
    2. Petzold, L., 1983. Automatic selection of methods for solving stiff and
       nonstiff systems of ordinary differential equations. SIAM journal on
       scientific and statistical computing, 4(1), pp.136-148.

    """

    _supports = {'multi_initials': True, 'multi_param_values': True}

    _memory_options = {'global': '0', 'shared': '1', 'sharedconstant': '2'}

    default_integrator_options = {
        # some sane default options for a few well-known integrators
        'cupsoda': {
            'max_steps': 20000, # max # of internal iterations (LSODA's MXSTEP)
            'atol': 1e-8,  # absolute tolerance
            'rtol': 1e-8,  # relative tolerance
            'n_blocks': None,  # number of GPU blocks
            'memory_usage': 'sharedconstant'}}  # see _memory_options dict

    _integrator_options_allowed = {'max_steps', 'atol', 'rtol', 'n_blocks',
                                   'memory_usage', 'vol'}

    def __init__(self, model, tspan=None, initials=None, param_values=None,
                 verbose=False, **kwargs):
        super(CupSodaSimulator, self).__init__(model, tspan=tspan,
                                               initials=initials,
                                               param_values=param_values,
                                               verbose=verbose, **kwargs)
        self.gpu = kwargs.pop('gpu', 0)
        self._obs_species_only = kwargs.pop('obs_species_only', True)
        self._cleanup = kwargs.pop('cleanup', True)
        self._prefix = kwargs.pop('prefix', self._model.name.replace('.', '_'))
        self._base_dir = kwargs.pop('base_dir', None)
        self.integrator = kwargs.pop('integrator', 'cupsoda')
        integrator_options = kwargs.pop('integrator_options', {})

        if kwargs:
            raise ValueError('Unknown keyword argument(s): {}'.format(
                ', '.join(kwargs.keys())
            ))

        unknown_integrator_options = set(integrator_options.keys()).difference(
            self._integrator_options_allowed
        )
        if unknown_integrator_options:
            raise ValueError(
                'Unknown integrator_options: {}. Allowed options: {}'.format(
                    ', '.join(unknown_integrator_options),
                    ', '.join(self._integrator_options_allowed)
                )
            )

        # generate the equations for the model
        pysb.bng.generate_equations(self._model, self._cleanup, self.verbose)

        # build integrator options list from our defaults and any kwargs
        # passed to this function
        options = {}
        if self.default_integrator_options.get(self.integrator):
            options.update(self.default_integrator_options[
                self.integrator])  # default options
        else:
            raise SimulatorException(
                "Integrator type '" + self.integrator + "' not recognized.")
        options.update(integrator_options)  # overwrite

        # defaults
        self.opts = options
        self._out_species = None

        # private variables (to reduce the number of function calls)
        self._len_rxns = len(self._model.reactions)
        self._len_species = len(self._model.species)
        self._len_params = len(self._model.parameters)
        self._model_parameters_rules = self._model.parameters_rules()

        # Set cupsoda verbosity level
        logger_level = self._logger.logger.getEffectiveLevel()
        if logger_level <= EXTENDED_DEBUG:
            self._cupsoda_verbose = 2
        elif logger_level <= logging.DEBUG:
            self._cupsoda_verbose = 1
        else:
            self._cupsoda_verbose = 0

        # regex for extracting cupSODA reported running time
        self._running_time_regex = re.compile(r'Running time:\s+(\d+\.\d+)')

    def run(self, tspan=None, initials=None, param_values=None):
        """Perform a set of integrations.

        Returns a :class:`.SimulationResult` object.

        Parameters
        ----------
        tspan : list-like, optional
            Time values at which the integrations are sampled. The first and
            last values define the time range.
        initials : list-like, optional
            Initial species concentrations for all simulations. Dimensions are
            number of simulation x number of species.    
        param_values : list-like, optional
            Parameters for all simulations. Dimensions are number of
            simulations x number of parameters.

        Returns
        -------
        A :class:`SimulationResult` object

        Notes
        -----
        1. An exception is thrown if `tspan` is not defined in either
           `__init__`or `run`.
           
        2. If neither `initials` nor `param_values` are defined in either 
           `__init__` or `run` a single simulation is run with the initial 
           concentrations and parameter values defined in the model.

        """
        super(CupSodaSimulator, self).run(tspan=tspan, initials=initials,
                                          param_values=param_values,
                                          _run_kwargs=[])

        # Create directories for cupSODA input and output files
        self.outdir = tempfile.mkdtemp(prefix=self._prefix + '_',
                                       dir=self._base_dir)

        self._logger.debug("Output directory is %s" % self.outdir)
        _cupsoda_infiles_dir = os.path.join(self.outdir, "INPUT")
        os.mkdir(_cupsoda_infiles_dir)
        self._cupsoda_outfiles_dir = os.path.join(self.outdir, "OUTPUT")

        # Path to cupSODA executable
        bin_path = get_path('cupsoda')

        # Create cupSODA input files
        self._create_input_files(_cupsoda_infiles_dir)

        # Build command
        # ./cupSODA input_model_folder blocks output_folder simulation_
        # file_prefix gpu_number fitness_calculation memory_use dump
        command = [bin_path, _cupsoda_infiles_dir, str(self.n_blocks),
                   self._cupsoda_outfiles_dir, self._prefix, str(self.gpu),
                   '0', self._memory_usage, str(self._cupsoda_verbose)]

        self._logger.info("Running cupSODA: " + ' '.join(command))
        start_time = time.time()
        # Run simulation and return trajectories
        p = subprocess.Popen(command, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        # for line in iter(p.stdout.readline, b''):
        #     if 'Running time' in line:
        #         self._logger.info(line[:-1])
        (p_out, p_err) = p.communicate()
        p_out = p_out.decode('utf-8')
        p_err = p_err.decode('utf-8')
        logger_level = self._logger.logger.getEffectiveLevel()
        if logger_level <= logging.INFO:
            run_time_match = self._running_time_regex.search(p_out)
            if run_time_match:
                self._logger.info('cupSODA reported time: {} '
                                  'seconds'.format(run_time_match.group(1)))
        self._logger.debug('cupSODA stdout:\n' + p_out)
        if p_err:
            self._logger.error('cupsoda strerr:\n' + p_err)
        if p.returncode:
            raise SimulatorException(
                p_out.rstrip("at line") + "\n" + p_err.rstrip())
        tout, trajectories = self._load_trajectories(
            self._cupsoda_outfiles_dir)
        if self._cleanup:
            shutil.rmtree(self.outdir)
        end_time = time.time()
        self._logger.info("cupSODA + I/O time: {} seconds".format(end_time -
                                                                  start_time))
        return SimulationResult(self, tout, trajectories)

    @property
    def _memory_usage(self):
        try:
            return self._memory_options[self.opts['memory_usage']]
        except KeyError:
            raise Exception('memory_usage must be one of %s',
                            self._memory_options.keys())

    @property
    def vol(self):
        vol = self.opts.get('vol', None)
        return vol

    @vol.setter
    def vol(self, volume):
        self.opts['vol'] = volume

    @property
    def n_blocks(self):
        n_blocks = self.opts.get('n_blocks')
        if n_blocks is None:
            default_threads_per_block = 32
            bytes_per_float = 4
            memory_per_thread = (self._len_species + 1) * bytes_per_float
            if cuda is None:
                threads_per_block = default_threads_per_block
            else:
                cuda.init()
                device = cuda.Device(self.gpu)
                attrs = device.get_attributes()
                shared_memory_per_block = attrs[
                    cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]
                upper_limit_threads_per_block = attrs[
                    cuda.device_attribute.MAX_THREADS_PER_BLOCK]
                max_threads_per_block = min(
                    shared_memory_per_block / memory_per_thread,
                    upper_limit_threads_per_block)
                threads_per_block = min(max_threads_per_block,
                                        default_threads_per_block)
            n_blocks = int(
                np.ceil(1. * len(self.param_values) / threads_per_block))
            self._logger.debug('n_blocks set to {} (used pycuda: {})'.format(
                n_blocks, cuda is not None
            ))
        self.n_blocks = n_blocks
        return n_blocks

    @n_blocks.setter
    def n_blocks(self, n_blocks):
        if not isinstance(n_blocks, int):
            raise ValueError("n_blocks must be an integer")
        if n_blocks <= 0:
            raise ValueError("n_blocks must be greater than 0")
        self.opts['n_blocks'] = n_blocks

    def _create_input_files(self, directory):

        n_sims = len(self.param_values)

        # atol_vector
        with open(os.path.join(directory, "atol_vector"), 'w') as atol_vector:
            for i in range(self._len_species):
                atol_vector.write(str(self.opts.get('atol')))
                if i < self._len_species - 1:
                    atol_vector.write("\n")

        # c_matrix
        with open(os.path.join(directory, "c_matrix"), 'w') as c_matrix:
            cmtx = self._get_cmatrix()
            for i in range(n_sims):
                line = ""
                for j in range(self._len_rxns):
                    if j > 0:
                        line += "\t"
                    line += str(cmtx[i][j])
                c_matrix.write(line)
                if i < n_sims - 1:
                    c_matrix.write("\n")

        # cs_vector
        with open(os.path.join(directory, "cs_vector"), 'w') as cs_vector:
            self._out_species = range(self._len_species)  # species to output
            if self._obs_species_only:
                self._out_species = [False for sp in self._model.species]
                for obs in self._model.observables:
                    for i in obs.species:
                        self._out_species[i] = True
                self._out_species = [i for i in range(self._len_species) if
                                     self._out_species[i] is True]
            for i in range(len(self._out_species)):
                if i > 0:
                    cs_vector.write("\n")
                cs_vector.write(str(self._out_species[i]))

        # left_side
        with open(os.path.join(directory, "left_side"), 'w') as left_side:
            for i in range(self._len_rxns):
                line = ""
                for j in range(self._len_species):
                    if j > 0:
                        line += "\t"
                    stoich = 0
                    for k in self._model.reactions[i]['reactants']:
                        if j == k:
                            stoich += 1
                    line += str(stoich)
                if i < self._len_rxns - 1:
                    left_side.write(line + "\n")
                else:
                    left_side.write(line)

        # max_steps
        with open(os.path.join(directory, "max_steps"), 'w') as mxsteps:
            mxsteps.write(str(self.opts['max_steps']))

        # model_kind
        with open(os.path.join(directory, "modelkind"), 'w') as model_kind:
            # always set modelkind to 'deterministic'
            model_kind.write("deterministic")

        # MX_0
        with open(os.path.join(directory, "MX_0"), 'w') as MX_0:
            mx0 = self.initials
            # if a volume has been defined, rescale populations
            # by N_A*vol to get concentration
            # (NOTE: act on a copy of self.initials, not
            # the original, which we don't want to modify)
            if self.vol:
                mx0 = mx0.copy()
                mx0 /= (N_A * self.vol)
            for i in range(n_sims):
                line = ""
                for j in range(self._len_species):
                    if j > 0:
                        line += "\t"
                    line += str(mx0[i][j])
                MX_0.write(line)
                if i < n_sims - 1:
                    MX_0.write("\n")

        # right_side
        with open(os.path.join(directory, "right_side"), 'w') as right_side:
            for i in range(self._len_rxns):
                line = ""
                for j in range(self._len_species):
                    if j > 0:
                        line += "\t"
                    stochiometry = 0
                    for k in self._model.reactions[i]['products']:
                        if j == k:
                            stochiometry += 1
                    line += str(stochiometry)
                if i < self._len_rxns - 1:
                    right_side.write(line + "\n")
                else:
                    right_side.write(line)

        # rtol
        with open(os.path.join(directory, "rtol"), 'w') as rtol:
            rtol.write(str(self.opts.get('rtol')))

        # t_vector
        with open(os.path.join(directory, "t_vector"), 'w') as t_vector:
            for t in self.tspan:
                t_vector.write(str(float(t)) + "\n")

        # time_max
        with open(os.path.join(directory, "time_max"), 'w') as time_max:
            time_max.write(str(float(self.tspan[-1])))

    def _get_cmatrix(self):
        self._logger.debug("Constructing the c_matrix:")
        c_matrix = np.zeros((len(self.param_values), self._len_rxns))
        par_names = [p.name for p in self._model_parameters_rules]
        rate_mask = np.array([p in self._model_parameters_rules for p in
                              self._model.parameters])
        rate_args = []
        par_vals = self.param_values[:, rate_mask]
        rate_order = []
        for rxn in self._model.reactions:
            rate_args.append([arg for arg in rxn['rate'].atoms(sympy.Symbol) if
                              not arg.name.startswith('__s')])
            reactants = len(rxn['reactants'])
            rate_order.append(reactants)
        output = 0.01 * len(par_vals)
        output = int(output) if output > 1 else 1
        for i in range(len(par_vals)):
            if i % output == 0:
                self._logger.debug(str(int(round(100. * i / len(par_vals)))) +
                                   "%")
            for j in range(self._len_rxns):
                rate = 1.0
                for r in rate_args[j]:
                    if isinstance(r, pysb.Parameter):
                        rate *= par_vals[i][par_names.index(r.name)]
                    elif isinstance(r, pysb.Expression):
                        raise ValueError('cupSODA does not currently support '
                                         'models with Expressions')
                    else:
                        rate *= r
                # volume correction
                if self.vol:
                    rate *= (N_A * self.vol) ** (rate_order[j] - 1)
                c_matrix[i][j] = rate
        self._logger.debug("100%")
        return c_matrix

    def _load_trajectories(self, directory):
        """Read simulation results from output files.

        Returns `tout` and `trajectories` arrays.
        """
        files = [filename for filename in os.listdir(directory) if
                 re.match(self._prefix, filename)]
        if len(files) == 0:
            raise SimulatorException(
                "Cannot find any output files to load data from.")
        if len(files) != len(self.param_values):
            raise SimulatorException(
                "Number of input files (%d) does not match number "
                "of requested simulations (%d)." % (
                len(files), len(self.param_values)))
        n_sims = len(files)
        trajectories = [None] * n_sims
        tout = [None] * n_sims
        traj_n = np.ones((len(self.tspan), self._len_species)) * float('nan')
        tout_n = np.ones(len(self.tspan)) * float('nan')
        # load the data
        indir_prefix = os.path.join(directory, self._prefix)
        for n in range(n_sims):
            trajectories[n] = traj_n.copy()
            tout[n] = tout_n.copy()
            filename = indir_prefix + "_" + str(n)
            if not os.path.isfile(filename):
                raise Exception("Cannot find input file " + filename)
            # determine optimal loading method
            if n == 0:
                (data, use_pandas) = self._test_pandas(filename)
            # load data
            else:
                if use_pandas:
                    data = self._load_with_pandas(filename)
                else:
                    data = self._load_with_openfile(filename)
            # store data
            tout[n] = data[:, 0]
            trajectories[n][:, self._out_species] = data[:, 1:]
            # volume correction
            if self.vol:
                trajectories[n][:, self._out_species] *= (N_A * self.vol)
        return np.array(tout), np.array(trajectories)

    def _test_pandas(self, filename):
        """ calculates the fastest method to load in data

        Parameters
        ----------
        filename : str
            filename to laod in

        Returns
        -------
        np.array, bool

        """
        # using open(filename,...)
        start = time.time()
        data = self._load_with_openfile(filename)
        end = time.time()
        load_time_openfile = end - start

        # using pandas
        if pd:
            start = time.time()
            self._load_with_pandas(filename)
            end = time.time()
            load_time_pandas = end - start
            if load_time_pandas < load_time_openfile:
                return data, True

        return data, False

    @staticmethod
    def _load_with_pandas(filename):
        data = pd.read_csv(filename, sep='\t', skiprows=None,
                           header=None).as_matrix()
        return data

    @staticmethod
    def _load_with_openfile(filename):
        with open(filename, 'r') as f:
            data = [line.rstrip('\n').split() for line in f]
        data = np.array(data, dtype=np.float, copy=False)
        return data


def run_cupsoda(model, tspan, initials=None, param_values=None,
                integrator='cupsoda', cleanup=True, verbose=False, **kwargs):
    """Wrapper method for running cupSODA simulations.
    
    Parameters
    ----------
    See ``CupSodaSimulator`` constructor.
    
    Returns
    -------
    SimulationResult.all : list of record arrays
        List of trajectory sets. The first dimension contains species,
        observables and expressions (in that order)
    """
    sim = CupSodaSimulator(model, tspan=tspan, integrator=integrator,
                           cleanup=cleanup, verbose=verbose, **kwargs)
    simres = sim.run(initials=initials, param_values=param_values)
    return simres.all
