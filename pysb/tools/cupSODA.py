import os
import warnings
import numpy
import scipy
import subprocess
import re
import pysb

_cupsoda_path = None

# Putting this here because the cupSODA class may be generalized in the future to a MultiSolver.
# All supported integrators should be included here, even if they don't have default options.
# Obviously, the only integrator currently supported is "cupSODA" (case insensitive).
default_integrator_options = {
    'cupsoda': {
        'gen_net': False,
        'atol': 1e-8,
        'rtol': 1e-8,
        'vol': None, 
        'n_blocks': 64, 
        'gpu': 0, 
        'outdir': os.getcwd(), 
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
    if _cupsoda_path:
        return _cupsoda_path

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
    integrator : string, optional
        Name of the integrator to use, taken from the integrators listed in 
        pysb.tools.cupSODA.default_integrator_options.
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
    model : pysb.Model
        Model passed to the constructor.
    tspan : vector-like
        Time values passed to the constructor.
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

    """
    
    def __init__(self, model, tspan, integrator='cupsoda', **integrator_options):

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
        if (len(model.reactions)==0 and len(model.species)==0) or options.get('gen_net')==True:
            pysb.bng.generate_equations(model)

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

    def run(self, prefix=None):
        
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
        N_SIMS = len(self.k)
        N_RXNS = len(self.k[0])
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
        #####FIXME
#         obs_species = [False for i in self.model.species]
        obs_species = [True for i in self.model.species]
        #####
        for obs in self.model.observables:
            for sp in obs.species:
                obs_species[sp] = True
        first = True
        for i in range(len(obs_species)):
            if (obs_species[i]):
                if not first: cs_vector.write("\n")
                else: first = False
                cs_vector.write(str(i))
        cs_vector.close()
        
        # left_side
        left_side = open(os.path.join(cupsoda_files,"left_side"), 'wb')
        for i in range(len(self.model.reactions)):
            line = ""
            for j in range(len(self.model.species)):
                if j > 0: line += "\t"
                stoich = 0
                for k in range(len(self.model.reactions[i]['reactants'])):
                    if j == self.model.reactions[i]['reactants'][k]:
                        stoich += 1
                line += str(stoich)
            if (i < len(self.model.reactions)-1): left_side.write(line+"\n")
            else: left_side.write(line)
        left_side.close()
        
        # modelkind
        modelkind = open(os.path.join(cupsoda_files,"modelkind"), 'wb')
        if not self.vol: modelkind.write("deterministic")
        else: modelkind.write("stochastic")
        modelkind.close()
        
        # MX_0
        N_SPECIES = len(self.y0[0])
        MX_0 = open(os.path.join(cupsoda_files,"MX_0"), 'wb')
        init = [0.0 for j in self.model.species]
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
                for k in range(len(self.model.reactions[i]['products'])):
                    if j == self.model.reactions[i]['products'][k]:
                        stoich += 1
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
        for i in range(len(self.tspan)):
            t_vector.write(str(float(self.tspan[i]))+"\n")
        t_vector.close()
        
        # time_max
        time_max = open(os.path.join(cupsoda_files,"time_max"), 'wb')
        time_max.write(str(float(self.tspan[len(self.tspan)-1])))
        time_max.close()
        
        # volume
        if (self.vol):
            warnings.warn("'vol' argument detected in run_cupSODA(): Make sure your rate constants and " +
                          "initial species amounts are in number units and that the population of the " +
                          "__source() species (if it exists) equals 6.0221413e23*vol.")
            volume = open(os.path.join(cupsoda_files,"volume"), 'wb')
            volume.write(str(self.vol))
            volume.close()
            
        # Build command
        command = [bin_path, cupsoda_files, str(self.n_blocks), self.outdir, prefix, str(self.gpu), str(0)]
        print "Running cupSODA: " + ' '.join(command)
        
        # Run simulation
#         subprocess.call(command)

        # Process data
        # ...
    
    