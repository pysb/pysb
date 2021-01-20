from __future__ import print_function
from pysb.simulator.base import SimulationResult
from pysb.bng import generate_equations
from pysb.simulator.ssa_base import SSABase
import numpy as np
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial




try:
    import pyopencl as cl
    from pyopencl import array as ocl_array
    from pyopencl import device_type
    from mako.template import Template
except ImportError:
    cl = None
    device_type = None
    ocl_array = None


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

        self.tspan = tspan
        self.verbose = verbose

        # private attribute
        self._step_0 = True
        self._dtype = precision

        template_code = Template(
            filename=os.path.join(os.path.dirname(__file__),
                                  'templates', 'opencl_ssa.cl')
        )

        args = self._get_template_args()
        # The use of single precision seems to work for most cases with little
        # differences in trajectory distributions, however sometimes if there
        # are really small time scale reactions, tau is below the level of
        # decimal precision time doesn't progress. Using double for time
        # seems to fix that for EARM 1.0, however, if there are reactions
        # who propensities make a0 too large, i think the no change in
        # time, perpetual running on gpu would occur. Keeping this warning
        # for now. For all other models that I have simulated, single produces
        # non-significantly different traces.
        if self._dtype == np.float32:
            args['prec'] = '#define USE_SINGLE_PRECISION'
            self._logger.warn("Should be cautious using single precision.")
        else:
            args['prec'] = '#define USE_DOUBLE_PRECISION'
        # So far, int 32 seems to handle all situations.
        # For boolean simulations, can use short or possible char.
        # Keeping option for reference now in hopes of being able to
        # calculate precision factor prior to simulating.
        _d = {np.uint32: "uint",
              np.int32: 'int',
              np.int16: 'ushort',
              np.int64: 'long',
              np.uint64: 'unsigned long'}
        self._dtype_species = np.int32
        args['spc_type'] = _d[self._dtype_species]
        if verbose == 2:
            args['verbose'] = '#define VERBOSE'
        elif verbose > 3:
            args['verbose'] = '#define VERBOSE_MAX'
        else:
            args['verbose'] = ''
        self._logger.info("Initialized OpenCLSSASimulator class")

        self._code = template_code.render(**args)

    def _compile(self):

        if self.verbose:
            self._logger.info("Output OpenCl file to ssa_opencl_code.cl")
            with open("ssa_opencl_code.cl", "w") as source_file:
                source_file.write(self._code)

        # allow users to select platform and devices
        self.context = cl.create_some_context(True)
        devices = self.context.devices
        self._device_name = devices[0].name
        self._n_devices = self.context.num_devices

        # set local work group size. CPU = 1, CUDA = 32, otherwise 64
        if devices[0].type == device_type.CPU:
            self._local_work_size = (1, 1)
        elif 'CUDA' in devices[0].platform.name.upper():
            self._local_work_size = (32, 1)
        else:
            self._local_work_size = (64, 1)

        self._logger.info(f"Using device {self._device_name}")
        self.devices = devices

        self.program = cl.Program(self.context, self._code).build(
            options=[
                # '-O3',
                # '-cl-std=2.0',
                # '-cl-uniform-work-group-size',
                # '-cl-fast-relaxed-math',
                # '-cl-single-precision-constant',
                '-cl-denorms-are-zero',
                '-cl-no-signed-zeros',
                '-cl-finite-math-only',
                '-cl-mad-enable',
                '-I {}'.format(os.path.join(os.path.dirname(__file__)))
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
        n_sim_per_device = self.num_sim//self._n_devices

        if self.num_sim < self._local_work_size[0]:
            local_work_size = (1, 1)
        else:
            local_work_size = self._local_work_size
        blocks, threads = self.get_blocks(n_sim_per_device,
                                          local_work_size[0])
        total_threads = int(blocks * threads)

        # retrieve and store results
        timer_start = time.time()

        with ThreadPoolExecutor(self._n_devices) as executor:
            sim_partial = partial(call, ocl_instance=self,
                                  n_sim_per_device=n_sim_per_device,
                                  t_out=t_out,
                                  total_threads=total_threads,
                                  local_work_size=local_work_size)
            results = [executor.submit(sim_partial, i)
                       for i in range(self._n_devices)]
            try:
                traj = [
                    r.result().reshape(
                        (total_threads, len(t_out), self._n_species)
                    )
                    for r in results
                ]
            finally:
                for r in results:
                    r.cancel()

        traj = np.vstack(traj)
        traj = traj[:self.num_sim]
        self._time = time.time() - timer_start
        self._logger.info("{} simulations "
                          "in {:.4f}s".format(self.num_sim, self._time))

        tout = np.array([tspan] * self.num_sim)
        return SimulationResult(self, tout, traj)

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


def call(device_number, ocl_instance, n_sim_per_device, t_out, total_threads,
         local_work_size):
    """ Function that allows multiple-devices function calls """

    # used to split entire simulation array to equal sizes per device
    start = device_number*n_sim_per_device
    end = device_number+n_sim_per_device

    # create command queue per device
    with cl.CommandQueue(ocl_instance.context,
                         ocl_instance.devices[device_number]) as queue:

        # transfer the array of time points to the device
        time_points_gpu = ocl_array.to_device(
            queue,
            np.array(t_out, dtype=ocl_instance._dtype)
        )
        # total_threads = int(self.num_sim)
        # transfer the array of seeds to the device
        random_seeds_gpu = ocl_array.to_device(
            queue,
            np.array(random.sample(range(2 ** 32 - 1), total_threads),
                     dtype=np.uint32)
        )

        # transfer the data structure of
        # ( number of simulations x different parameter sets )
        # to the device
        param_array_gpu = ocl_array.to_device(
            queue,
            _create_gpu_array(
                ocl_instance.param_values[start:end], total_threads,
                ocl_instance._dtype).flatten(order='C')
        )

        species_matrix_gpu = ocl_array.to_device(
            queue,
            _create_gpu_array(ocl_instance.initials[start:end],
                              total_threads,
                              ocl_instance._dtype_species).flatten(order='C')
        )

        result_gpu = ocl_array.zeros(
            queue,
            order='C',
            shape=(total_threads * t_out.shape[0] * ocl_instance._n_species,),
            dtype=ocl_instance._dtype_species
        )

        global_work_size = (total_threads, 1)

        ocl_instance._logger.debug(
            f"""Starting {ocl_instance.num_sim} simulations with 
            {global_work_size} workers and {ocl_instance._local_work_size}
            steps"""
        )
        # perform simulation
        complete_event = ocl_instance.program.Gillespie_all_steps(
            queue,
            global_work_size,
            local_work_size,
            species_matrix_gpu.data,
            result_gpu.data,
            time_points_gpu.data,
            param_array_gpu.data,
            random_seeds_gpu.data,
            np.int32(len(t_out)),
            np.int32(ocl_instance.num_sim)
        )

        complete_event.wait()
        return result_gpu.get(queue, async_=True)


def _create_gpu_array(values, total_threads, prec):

    # Create species matrix on GPU
    # will make according to number of total threads, not n_simulations
    gpu_array = np.zeros((total_threads, values.shape[1]), dtype=prec)
    # Filling species matrix
    # Note that this might not fill entire array that was created.
    # The rest of the array will be zeros to fill up GPU.
    gpu_array[:len(values)] = values
    return gpu_array