from pysb.simulate import Simulator
from pysb.bng import generate_equations
import numpy as np
from scipy.constants import N_A
import os
import warnings
import subprocess
import re
import itertools

_cupsoda_path = None

# Putting this here because the cupSODA class may be generalized in the future to a MultiSolver.
# All supported integrators should be included here, even if they don't have default options.
# Obviously, the only integrator currently supported is "cupSODA" (case insensitive).
default_integrator_options = {
    'cupsoda': {
        'max_steps': 20000,          # max no. of internal iterations (LSODA's MXSTEP)
        'atol': 1e-8,               # absolute tolerance
        'rtol': 1e-8,               # relative tolerance
        'n_blocks': None,           # number of GPU blocks
        'gpu': 0,                   # which GPU
        'vol': None,                # volume
        'obs_species_only': True,   # print only observable species
        'load_ydata': True,         # load conc data after simulation
        'memory_usage': 0,          # memory usage type: 
                                    # 0: global memory (default; suggested for medium-large models)
                                    # 1: shared memory 
                                    # 2: both shared and constant memory
        },
    }

def set_cupSODA_path(dir):
    global _cupsoda_path
    _cupsoda_path = os.path.join(dir,'cupSODA')
    # Make sure file exists and that it is not a directory
    if not os.access(_cupsoda_path, os.F_OK) or not os.path.isfile(_cupsoda_path):
        raise Exception('Could not find cupSODA binary in ' + os.path.abspath(dir) + '.')
    # Make sure file has executable permissions
    elif not os.access(_cupsoda_path, os.X_OK):
        raise Exception("cupSODA binary in " + os.path.abspath(dir) + " does not have executable permissions.")

def _get_cupSODA_path():
    """
    Return the path to the cupSODA executable.

    Looks for the cupSODA executable in a few hard-coded standard locations
    or in a user-defined location set via ``set_cupSODA_path``.

    """
    global _cupsoda_path
    
    # Just return cached value if it's available
    if _cupsoda_path: return _cupsoda_path

    bin_dirs = [
        '/usr/local/share/',
        'c:/Program Files/',
        ]

    def check_bin_dir(bin_dir):
        # Return the full path to the cupSODA executable or False if it
        # can't be found in one of the expected places.
        bin_path = os.path.join(bin_dir,'cupSODA')
        if os.access(bin_path, os.F_OK):
            return bin_path
        else:
            return False

    # Check the standard locations for the executable
    for b in bin_dirs:
        if check_bin_dir(b):
            break
        else:
            raise Exception('Could not find cupSODA installed in one of the following locations:' + 
                            ''.join('\n    ' + x for x in bin_dirs) + '\nPlease put the executable ' +
                            '(or a link to it) in one of these locations or set the path using set_cupSODA_path().')
            
    # Cache path for future use
    _cupsoda_path = bin_path
    return bin_path

class CupSODASolver(Simulator):
    """An interface for running cupSODA, a CUDA implementation of LSODA.
    Parameters
    ----------
    model : pysb.Model
        Model to integrate.
    tspan : vector-like, optional (default: None)
        Time values at which the integrations are sampled. The first and last values 
        define the time range.
    cleanup : bool, optional
        If True (default), delete the temporary files after the simulation is
        finished. If False, leave them in place. Useful for debugging.
    verbose : bool, optional (default: False)
        Verbose output 
    integrator : string, optional (default: 'cupsoda')
        Name of the integrator to use, taken from the integrators listed in 
        pysb.tools.cupSODA.default_integrator_options.
    integrator_options :
        * max_steps        : max no. of internal iterations (LSODA's MXSTEP) (default: None)
        * atol             : absolute tolerance (default: 1e-8)
        * rtol             : relative tolerance (default: 1e-8)
        * n_blocks         : number of GPU blocks (default: 64)
        * gpu              : index of GPU to run on (default: 0)
        * vol              : volume (required if number units; default: None)
        * obs_species_only : print only the concentrations of species in observables 
                             (default: True)
        * load_ydata       : read species concentrations from output files after simulation 
                             (default: True)
        * memory_usage     : type of memory usage (default: 0)
                             * 0: global memory (suggested for medium-large models)
                             * 1: shared memory 
                             * 2: both shared and constant memory

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
        An array view (sharing the same data buffer) on ``yexpr``. Dimensionality
        is ``(n_sims, len(tspan), len(model.expressions_dynamic()))``.
    outdir : string, optional (default: os.getcwd())
            Output directory.

    Notes
    -----
    1) The expensive step of generating the reaction network is performed during 
       initialization only if the network does not already exist (len(model.reactions)==0 
       and len(model.species)==0) OR if it is explicitly requested (gen_net==True).
    2) The cupSODA class is derived from the Solver class in pysb.integrate although 
       it overrides all of the functions of the Solver class (i.e, __init__() and run()). 
       This was done with the idea that eventually a lower level BaseSolver class will 
       be created from which the Solver and cupSODA classes will be derived. This will 
       probably be necessary for performing multiscale simulations where one would want 
       to embed a (potentially different) solver within each entity (e.g., cell) in the 
       simulation environment. Note that this is also why the argument list for the 
       __init__() function is identical to that of the Solver class.
    3) Like the Solver class, the cupSODA class accepts an optional 'integrator' string 
       argument. However, if anything other than 'cupsoda' (case insensitive) is passed 
       an Exception is thrown. The idea behind this design is that the cupSODA class may 
       eventually be generalized to a MultiSolver class, for which cupSODA is just one 
       of multiple supported integrators. This is analogous to the Solver class, which 
       currently supports all of the integrators included in :py:class:`scipy.integrate.ode`.
    4) The arrays 'y', 'yobs', and 'yobs_view' from the pysb.integrate.Solver base class
       have been overridden as dictionaries in which the keys are simulation numbers (ints)
       and the values are the same ndarrays as in the base class. For efficiency, these
       objects are initialized as empty dictionaries and only filled if the 'load_conc_data'
       argument to cupSODA.run() is set to True (the default) or if the cupSODA.load_data() 
       method is called directly.
    """
    
    def __init__(self, model, tspan=None, cleanup=True, verbose=False, integrator='cupsoda', **integrator_options):
        super(CupSODASolver, self).__init__(model, tspan, verbose)
        generate_equations(self.model, cleanup, self.verbose)

        # Set integrator options to defaults
        self.options = {}
        integrator = integrator.lower() # case insensitive
        if default_integrator_options.get(integrator.lower()):
            self.options.update(default_integrator_options[integrator])
        else:
            raise Exception("Integrator type '" + integrator + "' not recognized.")
        
        # overwrite default integrator options
        self.options.update(integrator_options)   

    def run(self, param_values, y0, tspan=None, outdir=os.getcwd(), prefix=None, **integrator_options):
        """Perform a set of integrations.

        Returns nothing; if load_ydata=True, can access the object's 
        ``y``, ``yobs``, or ``yobs_view`` attributes to retrieve the results.

        Parameters
        ----------
        param_values : list-like
            Rate constants for all reactions for all simulations. Dimensions are 
            ``(N_SIMS, len(model.reactions))``.
        y0 : list-like
            Initial species concentrations for all reactions for all simulations. 
            Dimensions are ``(N_SIMS, len(model.species))``.
        tspan : vector-like, optional
            Time values (exception thrown if set to None both here and in constructor).
        outdir : string, optional (default: os.getcwd())
            Output directory.
        prefix : string, optional (default: model.name)
            Output files will be named "prefix_i", for i=[0,N_SIMS).
        integrator_options : See CupSODASolver constructor.
        
        Notes
        -----
        If 'vol' is provided, cupSODA will assume that species counts are in number units 
        and will automatically convert them to concentrations by dividing by N_A*vol 
        (N_A = Avogadro's number).
        
        If load_ydata=True and obs_species_only=True, concentrations of species not in 
        observables are set to 'nan'.
        """
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
            raise Exception("Lengths of 'param_values' (len=" + str(len(param_values)) +") and 'y0' (len=" + str(len(y0)) + ") must be equal.")
        if len(param_values.shape) != 2 or param_values.shape[1] != len(self.model.reactions):
            raise Exception("'param_values' must be a 2D array with dimensions N_SIMS x N_RXNS: param_values.shape=" + str(self.param_values.shape))
        if len(y0.shape) != 2 or y0.shape[1] != len(self.model.species):
            raise Exception("'y0' must be a 2D array with dimensions N_SIMS x N_SPECIES: y0.shape=" + str(y0.shape))
        
        # Create outdir if it doesn't exist
        self.outdir = outdir
        if not os.path.exists(self.outdir): 
            os.makedirs(self.outdir)

        # Default prefix is model name
        if not prefix: 
            prefix = self.model.name
        
        # Path to cupSODA executable
        bin_path = _get_cupSODA_path() 
    
        # Put the cupSODA input files in a directory within the output directory
        cupsoda_files = os.path.join(self.outdir,"__CUPSODA_FILES")
        if os.path.exists(cupsoda_files):
            for f in os.listdir(cupsoda_files):
                os.remove(os.path.join(cupsoda_files,f))
        else:
            os.mkdir(cupsoda_files)
        
        # Simple default for number of blocks
        n_blocks = self.options.get('n_blocks')
        if not n_blocks:
            shared_memory_per_block = 48*1024 # bytes
            default_threads_per_block = 32
            n_species = len(self.model.species)
            bytes_per_float = 8
            memory_per_thread = (n_species+1)*bytes_per_float # +1 for time variable
            upper_limit_threads_per_block = 512
            max_threads_per_block = min( shared_memory_per_block/memory_per_thread , upper_limit_threads_per_block )
            threads_per_block = min( max_threads_per_block , default_threads_per_block )
            n_sims = param_values.shape[0]
            n_blocks = int(np.ceil(1.*n_sims/threads_per_block))
        
        # Create cupSODA input files
        self._create_input_files(cupsoda_files, param_values, y0)
            
        # Build command
        # ./cupSODA input_model_folder blocks output_folder simulation_file_prefix gpu_number fitness_calculation memory_use dump        
        command = [bin_path, cupsoda_files, str(n_blocks), self.outdir, prefix, str(self.options['gpu']), \
                   '0', str(self.options['memory_usage']), '1' if self.verbose else '0']
        print "Running cupSODA: " + ' '.join(command)
        
        # Run simulation
        subprocess.call(command)

        # Load concentration data if requested   
        if self.options.get('load_ydata'):
            self._get_y(prefix=prefix)
            self._calc_yobs_yexpr(param_values)
    
    def _create_input_files(self, cupsoda_files, param_values, y0):
        
        # Number of sims, rxns, and species
        n_sims,n_rxns = param_values.shape
        n_species = len(y0[0])
        
        # atol_vector 
        atol_vector = open(os.path.join(cupsoda_files,"atol_vector"), 'wb')
        for i in range(len(self.model.species)):
            atol_vector.write(str(self.options.get('atol')))
            if i < len(self.model.species)-1: atol_vector.write("\n")
        atol_vector.close()
        
        # c_matrix
        c_matrix = open(os.path.join(cupsoda_files,"c_matrix"), 'wb')
        for i in range(n_sims):
            line = ""
            for j in range(n_rxns):
                if j > 0: line += "\t"
                line += str(param_values[i][j])
            c_matrix.write(line)
            if i < n_sims-1: c_matrix.write("\n")
        c_matrix.close()
        
        # cs_vector
        cs_vector = open(os.path.join(cupsoda_files,"cs_vector"), 'wb')
        out_species = range(len(self.model.species))
        if self.options.get('obs_species_only'):
            out_species = [False for sp in self.model.species]
            for obs in self.model.observables:
                for i in obs.species:
                    out_species[i] = True
            out_species = [i for i in range(len(out_species)) if out_species[i]]
        for i in range(len(out_species)):
            if i > 0: cs_vector.write("\n")
            cs_vector.write(str(out_species[i]))
        cs_vector.close()
                
        # left_side
        left_side = open(os.path.join(cupsoda_files,"left_side"), 'wb')
        for i in range(len(self.model.reactions)):
            line = ""
            for j in range(len(self.model.species)):
                if j > 0: line += "\t"
                stoich = 0
                for k in self.model.reactions[i]['reactants']:
                    if j == k: stoich += 1
                line += str(stoich)
            if (i < len(self.model.reactions)-1): left_side.write(line+"\n")
            else: left_side.write(line)
        left_side.close()
        
        # max_steps
        mxsteps = open(os.path.join(cupsoda_files,"max_steps"), 'wb')
        mxsteps.write(str(self.options['max_steps']))
        mxsteps.close()
        
        # modelkind
        modelkind = open(os.path.join(cupsoda_files,"modelkind"), 'wb')
        vol = self.options['vol']
        if not vol: 
            modelkind.write("deterministic")
        else: 
            modelkind.write("stochastic")
        modelkind.close()
        
        # volume
        if vol:
            volume = open(os.path.join(cupsoda_files,"volume"), 'wb')
            volume.write(str(vol))
            volume.close()
            # Set population of __source() to N_A*vol and warn the user
            warnings.warn("Number units detected in cupSODA.run(). Setting the population " +
                          "of __source() (if it exists) equal to %g*%g." % (N_A,vol))
            for i,sp in enumerate(self.model.species):
                if str(sp) == '__source()':
                    y0[:,i] = N_A*vol
                    break
        # MX_0
        MX_0 = open(os.path.join(cupsoda_files,"MX_0"), 'wb')
        for i in range(n_sims):
            line = ""
            for j in range(n_species):
                if j > 0: line += "\t"
                line += str(y0[i][j])
            MX_0.write(line)
            if i < n_sims-1: MX_0.write("\n")
        MX_0.close()
        
        # right_side
        right_side = open(os.path.join(cupsoda_files,"right_side"), 'wb')
        for i in range(len(self.model.reactions)):
            line = ""
            for j in range(len(self.model.species)):
                if j > 0: line += "\t"
                stoich = 0
                for k in self.model.reactions[i]['products']:
                    if j == k: stoich += 1
                line += str(stoich)
            if (i < len(self.model.reactions)-1): right_side.write(line+"\n")
            else: right_side.write(line)
        right_side.close()
        
        # rtol
        rtol = open(os.path.join(cupsoda_files,"rtol"), 'wb')
        rtol.write(str(self.options.get('rtol')))
        rtol.close()
        
        # t_vector
        t_vector = open(os.path.join(cupsoda_files,"t_vector"), 'wb')
        t_vector.write("\n")
        for t in self.tspan:
            t_vector.write(str(float(t))+"\n")
        t_vector.close()
        
        # time_max
        time_max = open(os.path.join(cupsoda_files,"time_max"), 'wb')
        time_max.write(str(float(self.tspan[-1])))
        time_max.close() 
    
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
            Indices of species present in the data file (e.g., in cupSODA.run(), if
            obs_species_only=True then only species involved in observables are output to 
            file). If not specified, an attempt will be made to read them from the 
            'cs_vector' file in 'indir/__CUPSODA_FILES/'.
        """
        if indir is None: 
            indir = self.outdir
        if prefix is None: 
            prefix = self.model.name
        if out_species is None:
            try:
                out_species = np.loadtxt(os.path.join(indir,"__CUPSODA_FILES","cs_vector"), dtype=int)
            except IOError:
                print "ERROR: Cannot determine which species have been printed to file. Either provide an"
                print "'out_species' array or place the '__CUPSODA_FILES' directory in the input directory."
                raise
        
        FILES = [file for file in os.listdir(indir) if re.match(prefix, file)]
        if len(FILES)==0: 
            raise Exception("Cannot find any output files to load data from.")
        
        self.y = len(FILES) * [None]
        self.tout = len(FILES) * [None]
        for n in range(len(self.y)):
            self.y[n] = np.ones((len(self.tspan), len(self.model.species)))*float('nan')
            self.tout[n] = np.ones(len(self.tspan))*float('nan')
            # Read data from file
            file = os.path.join(indir,prefix)+"_"+str(n)
            if not os.path.isfile(file):
                raise Exception("Cannot find input file "+file)
            if self.verbose: 
                print "Reading "+file+" ...",
            f = open(file, 'rb')
            for i in range(len(self.y[n])):
                data = f.readline().split()
                self.tout[n][i] = data[0]
                self.y[n][i,out_species] = data[1:] # exclude first column (time)
            f.close()
            if self.verbose: print "Done."
#             if self.integrator.t < self.tspan[-1]: # NOT SURE IF THIS IS AN ISSUE HERE OR NOT
#                 self.y[i:, :] = 'nan'
        self.tout = np.array(self.tout)
        self.y = np.array(self.y)
        
    def _calc_yobs_yexpr(self, param_values=None):
        super(CupSODASolver, self)._calc_yobs_yexpr()
        
    def get_yfull(self):
        return super(CupSODASolver, self).get_yfull()

def run_cupsoda(model, tspan, param_values, y0, outdir=os.getcwd(), prefix=None, verbose=False, **integrator_options):

    sim = CupSODASolver(model, verbose=verbose, **integrator_options)
    sim.run(param_values, y0, tspan, outdir, prefix)
    if sim.options.get('load_ydata'):
        yfull = sim.get_yfull()
        return sim.tout, yfull
