from __future__ import print_function

import random

try:
    import pyopencl as cl
    from pyopencl import array as ocl_array
    from pyopencl import device_type
except ImportError:
    cl = None
import numpy as np
import os
import time
from pysb.bng import generate_equations
from pysb.simulator.base import SimulationResult, SimulatorException
from pysb.simulator.cuda_ssa import SSABase


class OpenCLSimulator(SSABase):
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
    device : str
        {'cpu', 'gpu'}
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

    def __init__(self, model, verbose=False, tspan=None, device='gpu',
                 **kwargs):

        if cl is None:
            raise SimulatorException('pyopencl library required for {}'
                                     ''.format(self.__class__.__name__))
        super(OpenCLSimulator, self).__init__(model, verbose, **kwargs)

        generate_equations(self._model)

        self.tout = None
        self.tspan = tspan
        self.verbose = verbose

        # private attribute
        self._parameter_number = len(self._model.parameters)
        self._n_species = len(self._model.species)
        self._n_reactions = len(self._model.reactions)
        self._step_0 = True
        template_code = _load_template()
        self._code = template_code.format(**self._get_template_args())
        if device is None:
            device = 'gpu'
        if device not in ('gpu', 'cpu'):
            raise AssertionError("device arg must be 'cpu' or 'gpu'")
        self._device = device
        self._logger.info("Initialized OpenCL class")

    def _compile(self):

        if self.verbose:
            self._logger.info("Output OpenCl file to ssa_opencl_code.cl")
            with open("ssa_opencl_code.cl", "w") as source_file:
                source_file.write(self._code)
        # This prints off all the options per device and platform
        self._logger.debug("Platforms availables")
        devices = []
        to_device = {'cpu': device_type.CPU, 'gpu': device_type.GPU}
        for i in cl.get_platforms():
            if len(i.get_devices(device_type=to_device[self._device])) > 0:
                devices = i.get_devices(device_type=to_device[self._device])
            self._logger.debug("\t{}\n\tDevices available".format(i))
            for j in i.get_devices():
                self._logger.debug("\t\t{}".format(j))
        # need to let the users select this
        self.context = cl.Context(devices)
        self.queue = cl.CommandQueue(self.context)
        self.program = cl.Program(self.context, self._code).build()

    def run(self, tspan=None, param_values=None, initials=None, number_sim=0):
        num_sim = int(number_sim)
        super(OpenCLSimulator, self).run(tspan=tspan, initials=initials,
                                         param_values=param_values,
                                         number_sim=num_sim)

        # tspan for each simulation
        t_out = np.array(tspan, dtype=np.float64)
        # compile kernel and send parameters to GPU
        if self._step_0:
            self._setup()

        self._logger.info("Starting {} simulations".format(number_sim))
        timer_start = time.time()

        # transfer the array of time points to the device
        time_points_gpu = ocl_array.to_device(
            self.queue,
            np.array(t_out, dtype=np.float64)
        )

        # transfer the array of time points to the device
        random_seeds_gpu = ocl_array.to_device(
            self.queue,
            np.array(random.sample(range(2 ** 32), num_sim))
        )

        # transfer the data structure of
        # ( number of simulations x different parameter sets )
        # to the device
        param_array_gpu = ocl_array.to_device(
            self.queue,
            self.param_values.astype(np.float64).flatten(order='C')
        )

        species_matrix_gpu = ocl_array.to_device(
            self.queue,
            self.initials.astype(np.int64).flatten(order='C')
        )

        result_gpu = ocl_array.zeros(
            self.queue,
            order='C',
            shape=(num_sim * len(t_out) * self._n_species,),
            dtype=np.int64
        )

        # perform simulation
        complete_event = self.program.Gillespie_all_steps(
            self.queue,
            (num_sim, 1,),
            None,
            species_matrix_gpu.data,
            result_gpu.data,
            time_points_gpu.data,
            param_array_gpu.data,
            random_seeds_gpu.data,
            np.int64(len(t_out)),
        )
        complete_event.wait()
        self._time = time.time() - timer_start
        self._logger.info("{} simulations "
                          "in {}s".format(number_sim, self._time))

        # retrieve and store results, only keeping n_simulations
        # actual simulations we will return
        tout = np.array([tspan] * num_sim)
        res = result_gpu.get(self.queue)
        res = res.reshape((num_sim, len(t_out), self._n_species))

        return SimulationResult(self, tout, res)

    def _setup(self):
        self._compile()
        self._step_0 = False


def _load_template():
    with open(os.path.join(os.path.dirname(__file__),
                           'pycuda_templates',
                           'opencl_template.cl'), 'r') as f:
        gillespie_code = f.read()
    return gillespie_code
