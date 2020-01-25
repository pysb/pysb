from __future__ import print_function
import random
import numpy as np
import os
import time
import warnings
try:
    import pyopencl as cl
    from pyopencl import array as ocl_array
    from pyopencl import device_type
except ImportError:
    cl = None
    device_type = None
    ocl_array = None

from pysb.bng import generate_equations
from pysb.simulator.base import SimulationResult
from pysb.simulator.cuda_ssa import SSABase


class OpenCLSSASimulator(SSABase):
    """
    OpenCL SSA simulator

    This simulator is capable of using either a GPU or multi-core CPU.
    The simulator will detect and ask which device you would like to use.
    Alteratively, you can set the device using with an environment variable
     `PYOPENCL_CTX`

    Requires `OpenCL`_ and `PyPpenCL`_.

    .. _OpenCL :
        https://www.khronos.org/opencl/
    .. _PyOpenCL :
        https://documen.tician.de/pyopencl/


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
    precision : (np.float64, np.float32)
        Precision for ssa simulation. Default is np.float64. float32 should
        be used with caution.

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

    def __init__(self, model, verbose=False, tspan=None, precision=np.float64,
                 **kwargs):

        if cl is None:
            raise ImportError('pyopencl library required for {}'
                              ''.format(self.__class__.__name__))
        super(OpenCLSSASimulator, self).__init__(model, verbose, **kwargs)

        generate_equations(self._model)

        self.tout = None
        self.tspan = tspan
        self.verbose = verbose

        # private attribute
        self._step_0 = True
        template_code = _load_template()
        self._code = template_code.format(**self._get_template_args())
        self._dtype = precision
        if self._dtype == np.float32:
            self._code = self._code.replace('double', 'float')
            self._code = self._code.replace('USE_DP', 'USE_FLOAT')
            warnings.warn("Should be cautious using single precision")
        if verbose == 2:
            self._code = self._code.replace('//#define VERBOSE',
                                            '#define VERBOSE')
        elif verbose > 3:
            self._code = self._code.replace('//#define VERBOSE',
                                            '#define VERBOSE_MAX')
        self._logger.info("Initialized OpenCLSSASimulator class")

    def _compile(self):

        if self.verbose:
            self._logger.info("Output OpenCl file to ssa_opencl_code.cl")
            with open("ssa_opencl_code.cl", "w") as source_file:
                source_file.write(self._code)

        # allow users to select platform and devices
        self.context = cl.create_some_context(True)
        devices = self.context.get_info(cl.context_info.DEVICES)
        # get platform of device (only one platform can be used, so using
        # first device will work )
        platform = devices[0].get_info(cl.device_info.PLATFORM)

        # check if a cpu, if so, we will change the work group size in run
        cpu_devices = platform.get_devices(device_type.CPU)
        use_cpu = len([i for i in cpu_devices if i in devices])

        # have not used FPGA but assumption is that it will require more
        # work, so assume gpu for now
        if use_cpu:
            self._local_work_size = (1, 1)
        # cuda uses a workgroup of 32, while amd/intel uses 64. Use in run
        elif 'CUDA' in platform.name.upper():
            self._local_work_size = (32, 1)
        else:
            self._local_work_size = (64, 1)

        self.queue = cl.CommandQueue(self.context, )

        self.program = cl.Program(self.context, self._code).build(
            options=[
                '-cl-no-signed-zeros',
                '-cl-mad-enable',
            ]
        )

    def run(self, tspan=None, param_values=None, initials=None, number_sim=0):
        """
        Run a simulation and returns the result (trajectories)

        .. note::
            In early versions of the Simulator class, ``tspan``, ``initials``
            and ``param_values`` supplied to this method persisted to future
            :func:`run` calls. This is no longer the case.

        Parameters
        ----------
        tspan
        initials
        param_values
            See parameter definitions in :class:`ScipyOdeSimulator`.
        number_sim: int
            Number of simulations to perform

        Returns
        -------
        A :class:`SimulationResult` object
        """
        super(OpenCLSSASimulator, self).run(tspan=tspan, initials=initials,
                                            param_values=param_values,
                                            number_sim=number_sim)
        if tspan is None:
            if self.tspan is None:
                raise Exception("Please provide tspan")
            else:
                tspan = self.tspan
        # tspan for each simulation
        t_out = np.array(tspan, dtype=self._dtype)
        # compile kernel and send parameters to GPU
        if self._step_0:
            self._setup()

        self._logger.info("Creating content on device")
        timer_start = time.time()

        # transfer the array of time points to the device
        time_points_gpu = ocl_array.to_device(
            self.queue,
            np.array(t_out, dtype=self._dtype)
        )
        if self.num_sim < self._local_work_size[0]:
            local_work_size = (1, 1)
        else:
            local_work_size = self._local_work_size

        blocks, threads = self.get_blocks(self.num_sim, local_work_size[0])
        total_threads = int(blocks * threads)
        # transfer the array of seeds to the device
        random_seeds_gpu = ocl_array.to_device(
            self.queue,
            np.array(random.sample(range(2 ** 32 - 1), self.num_sim),
                     dtype=np.uint32)
        )

        # transfer the data structure of
        # ( number of simulations x different parameter sets )
        # to the device
        param_array_gpu = ocl_array.to_device(
            self.queue,
            self._create_gpu_array(self.param_values, total_threads,
                                   self._dtype).flatten(order='C')
        )

        species_matrix_gpu = ocl_array.to_device(
            self.queue,
            self._create_gpu_array(self.initials, total_threads,
                                   np.uint32).flatten(order='C')
        )

        result_gpu = ocl_array.zeros(
            self.queue,
            order='C',
            shape=(total_threads * t_out.shape[0] * self._n_species,),
            dtype=np.uint32
        )

        elasped_t = time.time() - timer_start
        self._logger.info("Completed transfer in: {:.4f}s".format(elasped_t))
        global_work_size = (total_threads, 1)

        self._logger.debug("Starting {} simulations with {} workers "
                           "and {} steps".format(number_sim, global_work_size,
                                                 self._local_work_size))
        timer_start = time.time()

        # perform simulation
        complete_event = self.program.Gillespie_all_steps(
            self.queue,
            global_work_size,
            local_work_size,
            species_matrix_gpu.data,
            result_gpu.data,
            time_points_gpu.data,
            param_array_gpu.data,
            random_seeds_gpu.data,
            np.uint32(len(t_out)),
            np.uint32(self.num_sim)
        )
        complete_event.wait()
        self._time = time.time() - timer_start
        self._logger.info("{} simulations "
                          "in {:.4f}s".format(number_sim, self._time))

        # retrieve and store results
        timer_start = time.time()
        res = result_gpu.get(self.queue, async_=True)
        self._logger.info("Retrieved trajectories in {:.4f}s"
                          "".format(time.time() - timer_start))

        res = res.reshape((total_threads, len(t_out), self._n_species))
        res = res[:self.num_sim]
        tout = np.array([tspan] * self.num_sim)
        return SimulationResult(self, tout, res)

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
        max_tpb = 256
        if n_simulations < max_tpb:
            block_count = 1
            threads_per_block = max_tpb
        elif n_simulations % threads_per_block == 0:
            block_count = int(n_simulations // threads_per_block)
        else:
            block_count = int(n_simulations // threads_per_block + 1)
        return block_count, threads_per_block

    def _setup(self):
        self._compile()
        self._step_0 = False


def _load_template():
    _path = os.path.join(os.path.dirname(__file__), 'templates', 'opencl.cl')
    with open(_path, 'r') as f:
        gillespie_code = f.read()
    return gillespie_code
