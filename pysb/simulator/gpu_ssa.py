from __future__ import print_function

#
try:
    import pycuda
    import pycuda.autoinit
    import pycuda as cuda
    import pycuda.compiler
    import pycuda.tools as tools
    import pycuda.driver as driver
    import pycuda.gpuarray as gpuarray
except ImportError:
    pycuda = None

import re
import numpy as np
import os
import sympy
import time
from pysb.logging import setup_logger
import logging
from pysb.pathfinder import get_path
from pysb.core import Expression
from pysb.bng import generate_equations
from pysb.simulator.base import Simulator, SimulationResult, SimulatorException


class GPUSimulator(Simulator):
    """
    GPU simulator

    Parameters
    ----------
    model : pysb.Model
        Model to simulate.
    tspan : vector-like, optional
        Time values over which to simulate. The first and last values define
        the time range. Returned trajectories are sampled at every value unless
        the simulation is interrupted for some reason, e.g., due to
        satisfaction
        of a logical stopping criterion (see 'tout' below).
    initials : vector-like or dict, optional
        Values to use for the initial condition of all species. Ordering is
        determined by the order of model.species. If not specified, initial
        conditions will be taken from model.initial_conditions (with
        initial condition parameter values taken from `param_values` if
        specified).
    param_values : vector-like or dict, optional
        Values to use for every parameter in the model. Ordering is
        determined by the order of model.parameters.
        If passed as a dictionary, keys must be parameter names.
        If not specified, parameter values will be taken directly from
        model.parameters.
    verbose : bool, optional (default: False)
        Verbose output.

    Attributes
    ----------
    verbose: bool
        Verbosity flag passed to the constructor.
    model : pysb.Model
        Model passed to the constructor.
    tspan : vector-like
        Time values passed to the constructor.
    """
    _supports = {'multi_initials': True, 'multi_param_values': True}

    def __init__(self, model, verbose=False, tspan=None, **kwargs):
        super(GPUSimulator, self).__init__(model, verbose, **kwargs)

        if pycuda is None:
            raise SimulatorException('pycuda library required for {}'
                                     ''.format(self.__class__.__name__))

        generate_equations(self._model)

        self.tspan = tspan
        self.verbose = verbose

        # private attribute
        self._parameter_number = len(self._model.parameters)
        self._n_species = len(self._model.species)
        self._n_reactions = len(self._model.reactions)
        self._step_0 = True
        self._code = self._pysb_to_cuda()
        self._ssa_all = None
        self._kernel = None
        self._param_tex = None
        self._ssa = None

        if verbose:
            setup_logger(logging.INFO)
        self._logger.info("Initialized GPU class")

    def _pysb_to_cuda(self):
        """ converts pysb reactions to cuda compilable code

        """
        p = re.compile('\s')
        stoich_matrix = _get_stoch(self.model)

        all_reactions = []
        for rxn_number, rxn in enumerate(stoich_matrix.T):
            changes = []
            for index, change in enumerate(rxn):
                if change != 0:
                    changes.append([index, change])
            all_reactions.append(changes)

        params_names = [g.name for g in self._model.parameters]
        _reaction_number = len(self._model.reactions)

        stoich_string = ''
        l_lim = self._n_species - 1
        r_lim = self._n_reactions - 1

        for i in range(0, self._n_reactions):
            for j in range(0, len(stoich_matrix)):
                stoich_string += "%s" % repr(stoich_matrix[j][i])
                if not (i == l_lim and j == r_lim):
                    stoich_string += ','
            stoich_string += '\n'
        hazards_string = ''
        pattern = "(__s\d+)\*\*(\d+)"
        for n, rxn in enumerate(self._model.reactions):

            hazards_string += "\th[%s] = " % repr(n)
            rate = sympy.fcode(rxn["rate"])
            rate = re.sub('d0', '', rate)
            rate = p.sub('', rate)
            expr_strings = {
                e.name: '(%s)' % sympy.ccode(
                    e.expand_expr(expand_observables=True)
                ) for e in self.model.expressions}
            # expand only expressions used in the rate eqn
            for e in {sym for sym in rxn["rate"].atoms()
                      if isinstance(sym, Expression)}:
                rate = re.sub(r'\b%s\b' % e.name,
                              expr_strings[e.name],
                              rate)

            matches = re.findall(pattern, rate)
            for m in matches:
                repl = m[0]
                for i in range(1, int(m[1])):
                    repl += "*(%s-%d)" % (m[0], i)
                rate = re.sub(pattern, repl, rate)

            rate = re.sub(r'_*s(\d+)',
                          lambda m: 'y[%s]' % (int(m.group(1))),
                          rate)
            for q, prm in enumerate(params_names):
                rate = re.sub(r'\b(%s)\b' % prm, 'param_vec[%s]' % q, rate)
            items = rate.split('*')

            # create rate string. Places param_vec upfront to ensure that
            # the resulting value is a double (type of param_vec) and not
            # degraded to an int
            rate = [i for i in sorted(items) if i.startswith('param_vec')]
            rate += [i for i in sorted(items) if not i.startswith('param_vec')]
            rate = "*".join(rate)
            rate = re.sub('\*$', '', rate)
            rate = re.sub('d0', '', rate)
            rate = p.sub('', rate)
            rate = rate.replace('pow', 'powf')
            hazards_string += rate + ";\n"
        template_code = _load_template()
        cs_string = template_code.format(n_species=self._n_species,
                                         n_params=self._parameter_number,
                                         n_reactions=_reaction_number,
                                         hazards=hazards_string,
                                         stoch=stoich_string,
                                         )

        self._logger.debug("Converted PySB model to pycuda code")
        return cs_string

    def _compile(self):

        if self.verbose:
            self._logger.info("Output cuda file to ssa_cuda_code.cu")
            with open("ssa_cuda_code.cu", "w") as source_file:
                source_file.write(self._code)
        nvcc_bin = get_path('nvcc')
        self._logger.debug("Compiling CUDA code")
        opts = ['-O3', '--use_fast_math']
        self._kernel = pycuda.compiler.SourceModule(
            self._code, nvcc=nvcc_bin, options=opts, no_extern_c=True,
        )

        self._ssa = self._kernel.get_function("Gillespie_all_steps")
        self._logger.debug("Compiled CUDA code")
        self._step_0 = False

    def run(self, tspan=None, param_values=None, initials=None, number_sim=0,
            threads=32):

        num_sim = int(number_sim)

        # check for proper arguments
        if param_values is None and initials is None and not num_sim:
            raise InvalidSimulationValues()
        elif param_values is None and not num_sim:
            num_sim = np.array(initials).shape[0]
        elif initials is None and not num_sim:
            num_sim = np.array(param_values).shape[0]

        if param_values is None:
            # Run simulation using same param_values
            param_values = np.repeat(self.param_values, num_sim, axis=0)
        elif len(param_values.shape) == 1:
            param_values = np.repeat([param_values], num_sim, axis=0)

        if initials is None:
            # Run simulation using same initial conditions
            initials = np.repeat(self.initials, num_sim, axis=0)
        elif len(initials.shape) == 1:
            initials = np.repeat([initials], num_sim, axis=0)

        super(GPUSimulator, self).run(tspan=tspan, initials=initials,
                                      param_values=param_values,
                                      _run_kwargs=locals())

        if tspan is None:
            tspan = self.tspan

        tout = [tspan] * len(param_values)
        t_out = np.array(tspan, dtype=np.float64)

        # set default threads per block
        if threads is None:
            threads = 32

        blocks, threads = self.get_blocks(num_sim, threads)

        self._logger.info("Starting {} simulations on {} blocks"
                          "".format(num_sim, blocks))

        # compile kernel and send parameters to GPU
        if self._step_0:
            self._compile()
        if self.verbose:
            self._print_verbose(threads)

        #  Note, this number will be larger than n_simulations if the gpu grid
        #  is not filled. The rest will be filled with zeros and not impact
        #  results. They are trimmed right before passing to simulation results
        total_threads = int(blocks * threads)

        param_array_gpu = gpuarray.to_gpu(
            self._create_gpu_array(param_values, total_threads, np.float64)
        )

        species_matrix_gpu = gpuarray.to_gpu(
            self._create_gpu_array(initials, total_threads, np.int32)
        )

        # allocate and upload time to GPU
        time_points_gpu = gpuarray.to_gpu(np.array(t_out, dtype=np.float64))

        # allocate space on GPU for results
        result = driver.managed_zeros(
            shape=(total_threads, len(t_out), self._n_species),
            dtype=np.int32, mem_flags=driver.mem_attach_flags.GLOBAL
        )
        timer_start = time.time()
        # perform simulation
        self._ssa(species_matrix_gpu, result, time_points_gpu,
                  np.int32(len(t_out)), param_array_gpu,
                  block=(threads, 1, 1), grid=(blocks, 1))

        # Wait for kernel completion before host access
        pycuda.autoinit.context.synchronize()

        self._time = time.time() - timer_start
        self._logger.info("{} simulations "
                          "in {}s".format(num_sim, self._time))

        # retrieve and store results, only keeping num_sim (desired quantity)
        return SimulationResult(self, tout, result[:num_sim, :, :])

    def _print_verbose(self, threads):
        self._logger.debug("threads = {}".format(threads))
        kern = self._ssa
        self._logger.debug("Local memory  = {}".format(kern.local_size_bytes))
        self._logger.debug("Shared memory = {}".format(kern.shared_size_bytes))
        self._logger.debug("Registers  = {}".format(kern.num_regs))

        occ = tools.OccupancyRecord(tools.DeviceData(),
                                    threads=threads,
                                    shared_mem=kern.shared_size_bytes,
                                    registers=kern.num_regs)
        self._logger.debug("tb_per_mp  = {}".format(occ.tb_per_mp))
        self._logger.debug("limited by = {}".format(occ.limited_by))
        self._logger.debug("occupancy  = {}".format(occ.occupancy))
        self._logger.debug("tb/mp limits  = {}".format(occ.tb_per_mp_limits))

    @staticmethod
    def _create_gpu_array(values, total_threads, prec):

        # Create species matrix on GPU
        # will make according to number of total threads, not n_simulations
        gpu_array = np.zeros((total_threads, values.shape[1]), dtype=prec)
        # Filling species matrix
        # Note that this might not fill entire array that was created.
        # The rest of the array will be zeros to fill up GPU.
        gpu_array[:len(values)] = values
        return gpu_array

    @staticmethod
    def get_blocks(n_simulations, threads_per_block):
        max_threads = 256
        if threads_per_block > max_threads:
            logging.warning("Limit of 256 threads per block due to curand."
                            " Setting to 256.")
            threads_per_block = max_threads
        if n_simulations < max_threads:
            block_count = 1
            threads_per_block = max_threads
        elif n_simulations % threads_per_block == 0:
            block_count = int(n_simulations // threads_per_block)
        else:
            block_count = int(n_simulations // threads_per_block + 1)
        return block_count, threads_per_block


def _get_stoch(model):
    """
    Left hand side
    """
    left_side = np.zeros((len(model.reactions), len(model.species)),
                         dtype=np.int32)
    right_side = left_side.copy()

    for i in range(len(model.reactions)):
        for j in range(len(model.species)):
            stoich = 0
            for k in model.reactions[i]['reactants']:
                if j == k:
                    stoich += 1
            left_side[i, j] = stoich
            stoich = 0
            for k in model.reactions[i]['products']:
                if j == k:
                    stoich += 1
            right_side[i, j] = stoich
    return (right_side + left_side * -1).T


def _load_template():
    with open(os.path.join(os.path.dirname(__file__),
                           'pycuda_templates',
                           'gillespie_template.cu'), 'r') as f:
        gillespie_code = f.read()
    return gillespie_code


class InvalidSimulationValues(Exception):
    def __init__(self):
        Exception.__init__(self, "Please a multi-dimension set of parameters,"
                                 " initials, or number_sim>0")
