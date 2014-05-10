import os
import warnings
import numpy
import scipy
import subprocess
import re
import itertools
import pysb

_cupsoda_path = None

# Putting this here because the cupSODA class may be generalized in the future to a MultiSolver.
# All supported integrators should be included here, even if they don't have default options.
# Obviously, the only integrator currently supported is "cupSODA" (case insensitive).
default_integrator_options = {
    'cupsoda': {
        'gen_net': False,       # Force network generation?
        'atol': 1e-8,           # Absolute tolerance
        'rtol': 1e-8,           # Relative tolerance
        'vol': None,            # Volume
        'n_blocks': 64,         # Number of GPU blocks
        'gpu': 0,               # Which GPU
        'outdir': os.getcwd(),  # Output directory
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
    or in a user-defined location set via set_cupSODA_path().

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

class cupSODA(pysb.integrate.Solver):
    """An interface for parallel numerical integration of models.
    Parameters
    ----------
    model : pysb.Model
        Model to integrate.
    tspan : vector-like
        Time values at which the integrations are sampled. The first and last values 
        define the time range.
    integrator : string, optional (default: 'cupsoda')
        Name of the integrator to use, taken from the integrators listed in 
        pysb.tools.cupSODA.default_integrator_options.
    verbose : bool, optional (default: False)
        Verbose output 
    integrator_options :
        * k        : 2D list of rate constants with dimensions N_SIMS x N_RXNS 
                     (*REQUIRED)
        * y0       : 2D list of initial species concentrations with dimensions 
                     N_SIMS x N_SPECIES (*REQUIRED)
        * gen_net  : for models with already generated networks, force regeneration 
                     (default: False)
        * atol     : absolute tolerance (default: 1e-8)
        * rtol     : relative tolerance (default: 1e-8)
        * vol      : system volume (default: None)
        * n_blocks : number of GPU blocks (default: 64)
        * gpu      : index of GPU to run on (default: 0)
        * outdir   : output directory (default: os.getcwd())

    Attributes
    ----------
    verbose: bool
        Verbose output.
    model : pysb.Model
        Model passed to the constructor.
    tspan : vector-like
        Time values passed to the constructor.
    y : dict{ int : numpy.ndarray }
        Species trajectories for each simulation. Keys are simulation number,
        vals are arrays with dimensionality ``(len(tspan), len(model.species))``.
    yobs : dict{ int : numpy.ndarray (with record-style data-type) }
        Observable trajectories for each simulation. Keys are simulation number,
        vals are arrays with length ``len(tspan)``; record names follow 
        ``model.observables`` names.
    yobs_view : dict{ int : numpy.ndarray }
        Array views (sharing the same data buffer) on each array in ``yobs``. 
        Keys are simulation number, vals are arrays with dimensionality
        ``(len(tspan), len(model.observables))``.
    k : list-like
        Rate constants for all reactions for all simulations. Dimensions are 
        ``(N_SIMS, len(model.reactions))``.
    y0 : list-like
        Initial species concentrations for all reactions for all simulations. 
        Dimensions are ``(N_SIMS, len(model.species))``.
    atol : float
        Absolute tolerance.
    rtol : float
        Relative tolerance.
    vol : float
        System volume.
    n_blocks : int
        Number of GPU blocks.
    gpu : int
        Index of GPU to run on.
    outdir : string
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
    4) Currently, the run() function concludes once the simulations have finished, i.e., 
       no data is read in from the output files. This is only temporary and will 
       eventually be changed.
    5) The arrays 'y', 'yobs', and 'yobs_view' from the pysb.integrate.Solver base class
       have been overridden as dictionaries in which the keys are simulation numbers (ints)
       and the values are the same ndarrays as in the base class. For efficiency, these
       objects are initialized as empty dictionaries and only filled if the 'load_conc_data'
       argument to cupSODA.run() is set to True (the default) or if the cupSODA.load_data() 
       method is called directly.
    """
    
    def __init__(self, model, tspan, integrator='cupsoda', verbose=False, **integrator_options):

        self.verbose = verbose

        # Build integrator options list from our defaults and any kwargs passed to this function
        options = {}
        integrator = integrator.lower() # case insensitive
        if default_integrator_options.get(integrator.lower()):
            options.update(default_integrator_options[integrator]) # default options
        else:
            raise Exception("Integrator type '" + integrator + "' not recognized.")
        # overwrite defaults
        options.update(integrator_options)
        
        # Make sure 'k' and 'y0' have been defined
        if 'k' not in options.keys():
            raise Exception("2D array 'k' with dimensions N_SIMS x N_RXNS must be defined.")
        if 'y0' not in options.keys():
            raise Exception("2D array 'y0' with dimensions N_SIMS x N_SPECIES must be defined.")

        # Generate reaction network if it doesn't already exist or if explicitly requested
        if ( len(model.reactions)==0 and len(model.species)==0 ) or options.get('gen_net')==True:
            pysb.bng.generate_equations(model,self.verbose)

        # Set variables
        self.model = model
        self.tspan = tspan
        self.k = numpy.array(options.get('k'))
        self.y0 = numpy.array(options.get('y0'))
        self.atol = options.get('atol')
        self.rtol = options.get('rtol')
        self.vol = options.get('vol')
        self.n_blocks = options.get('n_blocks')
        self.gpu = options.get('gpu')
        self.outdir = options.get('outdir')

        # Error checks on 'k' and 'y0'
        if len(self.k) != len(self.y0):
            raise Exception("Lengths of 'k' (len=" + str(len(self.k)) +") and 'y0' (len=" + str(len(self.y0)) + ") must be equal.")
        if len(self.k.shape) != 2 or self.k.shape[1] != len(model.reactions):
            raise Exception("'k' must be a 2D array with dimensions N_SIMS x N_RXNS: k.shape=" + str(self.k.shape))
        if len(self.y0.shape) != 2 or self.y0.shape[1] != len(model.species):
            raise Exception("'y0' must be a 2D array with dimensions N_SIMS x N_SPECIES: y0.shape=" + str(self.y0.shape))
        
        # If outdir doesn't exist, create it
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
            
        # Initialize 'y', 'yobs', and 'yobs_view' as empty dictionaries
        self.y = {}
        self.yobs = {}
        self.yobs_view = {}

    def run(self, prefix=None, obs_species_only=True, load_conc_data=True):
        """Perform a set of integrations.

        Returns nothing; if load_conc_data=True, can access the object's 
        ``y``, ``yobs``, or ``yobs_view`` attributes to retrieve the results.

        Parameters
        ----------
        prefix : string, optional (default: model.name)
            Output files will be named "prefix_i", for i=[0,N_SIMS).
        obs_species_only : bool, optional (default: True)
            Specifies whether to print only the concentrations of species in observables. 
            A value of 'False' forces concentrations for *all* species to be printed.
        load_conc_data: bool, optional (default: True)
            Specifies whether species concentrations should be read in from output files 
            after simulation. If obs_species_only=True, concentrations of species not in 
            observables are set to 'nan'.
        """
        
        # Default prefix is model name
        if not prefix: prefix = self.model.name
        
        # Path to cupSODA executable
        bin_path = _get_cupSODA_path() 
    
        # Put the cupSODA input files in a directory within the output directory
        cupsoda_files = os.path.join(self.outdir,"__CUPSODA_FILES")
        if os.path.exists(cupsoda_files):
            for f in os.listdir(cupsoda_files):
                os.remove(os.path.join(cupsoda_files,f))
        else:
            os.mkdir(cupsoda_files)
            
        # Number of sims, rxns, and species
        N_SIMS,N_RXNS = self.k.shape
        N_SPECIES = len(self.y0[0])
        
        # Figure out number of blocks to run on
        # FIXME: Need to figure out how to calculate an optimal number
        if not self.n_blocks:
            self.n_blocks = 64
        
        # atol_vector 
        atol_vector = open(os.path.join(cupsoda_files,"atol_vector"), 'wb')
        for i in range(len(self.model.species)):
            atol_vector.write(str(self.atol))
            if i < len(self.model.species)-1: atol_vector.write("\n")
        atol_vector.close()
        
        # c_matrix
        c_matrix = open(os.path.join(cupsoda_files,"c_matrix"), 'wb')
        for i in range(N_SIMS):
            line = ""
            for j in range(N_RXNS):
                if j > 0: line += "\t"
                line += str(self.k[i][j])
            c_matrix.write(line)
            if i < N_SIMS-1: c_matrix.write("\n")
        c_matrix.close()
        
        # cs_vector
        cs_vector = open(os.path.join(cupsoda_files,"cs_vector"), 'wb')
        out_species = range(len(self.model.species))
        if obs_species_only:
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
        
        # modelkind
        modelkind = open(os.path.join(cupsoda_files,"modelkind"), 'wb')
        if not self.vol: modelkind.write("deterministic")
        else: modelkind.write("stochastic")
        modelkind.close()
        
        # volume
        if (self.vol):
            volume = open(os.path.join(cupsoda_files,"volume"), 'wb')
            volume.write(str(self.vol))
            volume.close()
            # Set population of __source() to 6.0221413e23*vol and warn the user
            warnings.warn("Number units detected in cupSODA.run(). Setting the population " +
                          "of __source() (if it exists) equal to 6.0221413e23*"+str(self.vol)+".")
            for i, s in enumerate(self.model.species):
                if str(s) == '__source()':
                    self.y0[:,i] = [6.0221413e23*self.vol for n in self.y0]
                    break
        
        # MX_0
        MX_0 = open(os.path.join(cupsoda_files,"MX_0"), 'wb')
        for i in range(N_SIMS):
            line = ""
            for j in range(N_SPECIES):
                if j > 0: line += "\t"
                line += str(self.y0[i][j])
            MX_0.write(line)
            if i < N_SIMS-1: MX_0.write("\n")
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
        rtol.write(str(self.rtol))
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
            
        # Build command
        command = [bin_path, cupsoda_files, str(self.n_blocks), self.outdir, prefix, str(self.gpu), str(0)]
        print "Running cupSODA: " + ' '.join(command)
        
        # Run simulation
        subprocess.call(command)

        # Load concentration data if requested   
        if load_conc_data: self.load_data()
            
    def load_data(self, indir=None, prefix=None, which_sims='ALL', out_species='ALL'):
        """Load simulation data from file.

        Returns nothing; fills the ``y``, ``yobs``, and ``yobs_view`` dictionary objects. 
        The keys are simulation numbers (ints) and the values are the same ndarrays as in
        the pysb.integrate.Solver base class.

        Parameters
        ----------
        indir : string, optional (default: self.outdir)
            Directory where data files are located.
        prefix : string, optional (default: model.name)
            Prefix for output filenames (e.g., "egfr" for egfr_0,...).
        which_sims : integer or list of integers, optional 
            Which data files to read concentration data from. If not specified, all data
            files present in 'indir' are read and the elements of 'which_sims' are based on 
            the names of the files (e.g., if the files are egfr_0, egfr_3, egfr_5, then 
            which_sims=[0,3,5]).
        out_species: integer or list of integers, optional 
            Indices of species present in the data file (e.g., in cupSODA.run(), if
            obs_species_only=True then only species involved in observables are output to 
            file). If not specified, an attempt will be made to read them from the 
            'cs_vector' file in 'indir/__CUPSODA_FILES/'. 
        """
                
        if not indir: indir = self.outdir
        if not prefix: prefix = self.model.name
        
        FILES = []
        if which_sims == 'ALL':
            FILES = [file for file in os.listdir(indir) if re.match(prefix, file)]
            if len(FILES)==0: 
                raise Exception("Cannot find any output files to load data from.")
        else:
            try:
                FILES = [prefix+"_"+str(which_sims[i]) for i in range(len(which_sims))] # list of integers
            except TypeError:
                FILES = [prefix+"_"+str(which_sims)] # single integer

        which_sims = [int(re.match(prefix+"_"+"(\d+)", file).group(1)) for file in FILES]
        
        if out_species == 'ALL':
            try:
                out_species = numpy.loadtxt(os.path.join(indir,"__CUPSODA_FILES","cs_vector"), dtype=int)
            except IOError:
                print "ERROR: Cannot determine which species have been printed to file. Either provide an"
                print "'out_species' array or place the '__CUPSODA_FILES' directory in the input directory."
                raise
                
        # Species concentrations and observables
        for sim in which_sims:
            # Create dictionaries: keys are sim numbers (ints), vals are ndarrays of concentrations
            self.y[sim] = numpy.ones((len(self.tspan), len(self.model.species)))*float('nan')
            if len(self.model.observables):
                self.yobs[sim] = numpy.empty(shape=len(self.tspan), dtype=zip(self.model.observables.keys(), itertools.repeat(float)))
            else:
                self.yobs[sim] = numpy.zeros((len(self.tspan), 0))
            self.yobs_view[sim] = self.yobs[sim].view(float).reshape((len(self.tspan), -1))

            # Read data from file
            file = os.path.join(indir,prefix)+"_"+str(sim)
            if not os.path.isfile(file):
                raise Exception("Cannot find input file "+file)
            if self.verbose: print "Reading "+file+" ...",
            f = open(file, 'rb')
            for i in range(len(self.y[sim])):
                self.y[sim][i,out_species] = f.readline().split()[1:] # exclude first column (time)
            f.close()
            if self.verbose: print "Done."
#             if self.integrator.t < self.tspan[-1]: # NOT SURE IF THIS IS AN ISSUE HERE OR NOT
#                 self.y[i:, :] = 'nan'
                
            # Calculate observables
            for j, obs in enumerate(self.model.observables):
                self.yobs_view[sim][:,j] = (self.y[sim][:,obs.species] * obs.coefficients).sum(axis=1)
