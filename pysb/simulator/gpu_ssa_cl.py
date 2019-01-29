from __future__ import print_function

#
import pyopencl as cl
from pyopencl import array
import re
import numpy as np
import os
import sympy
import time
from pysb.logging import setup_logger
import logging
from pysb.core import Expression
from pysb.bng import generate_equations
from pysb.simulator.base import Simulator, SimulationResult, SimulatorException


class GPUSimulatorCL(Simulator):
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
        super(GPUSimulatorCL, self).__init__(model, verbose, **kwargs)

        generate_equations(self._model)

        self.tout = None
        self.tspan = tspan
        self.verbose = verbose

        # private attribute
        self._threads = 32
        self._blocks = None
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
        stoich_matrix = (_rhs(self._model) + _lhs(self._model)).T

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
            rate = ''
            # places param_arry value at the front to make propensities doubles
            for i in items:
                if i.startswith('param_vec'):
                    rate += i + '*'

            for i in sorted(items):
                if i.startswith('param_vec'):
                    continue
                rate += i + '*'
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

    def _pysb_to_cuda2(self):
        """ converts pysb reactions to cuda compilable code

        """
        p = re.compile('\s')
        stoich_matrix = (_rhs(self._model) + _lhs(self._model)).T

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
        deps = dict()
        h_dict = dict()
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
            rate = ''
            # places param_arry value at the front to make propensities doubles
            for i in items:
                if i.startswith('param_vec'):
                    rate += i + '*'

            for i in sorted(items):
                if i.startswith('param_vec'):
                    continue
                rate += i + '*'
            rate = re.sub('\*$', '', rate)
            rate = re.sub('d0', '', rate)
            rate = p.sub('', rate)
            rate = rate.replace('pow', 'powf')

            hazards_string += rate + ";\n"
            deps[n] = list()
            for i in rxn['reactants']:
                deps[n].append(i)
                if i in h_dict:
                    h_dict[i].add("h[%s] = " % repr(n)+rate)
                else:
                    h_dict[i] = set(["h[%s] = " % repr(n)+rate])
            for i in rxn['products']:
                deps[n].append(i)
                if i in h_dict:
                    h_dict[i].add("h[%s] = " % repr(n)+rate)
                else:
                    h_dict[i] = set(["h[%s] = " % repr(n)+rate])

        final_string = "switch(rxn) {\n"
        for i in deps:
            h_string = '\tcase {} :\n'.format(i)
            to_add = set()
            for n in deps[i]:
                for m in h_dict[n]:
                    to_add.add(m)
            # print(to_add)
            for n in to_add:
                h_string += '\t\t{};\n'.format(n)
                new_line = n.replace('y', 'prev')
                new_line = re.sub(r'=', '-=', new_line)
                new_line = re.sub(r'h\[\d+]', 'a0', new_line)
                new_line2 = re.sub(r'=', '+=', n)
                new_line2 = re.sub(r'h\[\d+]', 'a0', new_line2)
                h_string += '\t\t{};\n'.format(new_line)
                h_string += '\t\t{};\n'.format(new_line2)
            # else:
            #     h_string += '\t\th[{}] =\t{};\n'.format(n, h_dict[n])
            h_string += "\t\tbreak;\n"
            # print(h_string)
            final_string += h_string
        final_string += '}'

        template_code = _load_template()

        cs_string = template_code.format(n_species=self._n_species,
                                         n_params=self._parameter_number,
                                         n_reactions=_reaction_number,
                                         hazards=hazards_string,
                                         stoch=stoich_string,
                                         hazards2=final_string)

        self._logger.debug("Converted PySB model to pycuda code")
        return cs_string

    def _compile(self, code):

        if self.verbose:
            self._logger.info("Output cuda file to ssa_cuda_code.cu")
            with open("ssa_cuda_code.cu", "w") as source_file:
                source_file.write(code)
        # This prints off all the options per device and platform
        print("Platforms availables")
        for i in cl.get_platforms():
            print("\t{}\n\tDevices available".format(i))
            for j in i.get_devices():
                print("\t\t{}".format(j))

        # need to let the users select this
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[1]
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        cache_dir = None
        if self.verbose:
            cache_dir = '.'

        self.program = cl.Program(self.context, self._code).build(
            cache_dir=cache_dir
        )

    def run(self, tspan=None, param_values=None, initials=None, number_sim=0,
            threads=32):

        if param_values is None:
            # Run simulation using same param_values
            num_particles = int(number_sim)
            nominal_values = np.array(
                [p.value for p in self._model.parameters])
            param_values = np.zeros((num_particles, len(nominal_values)),
                                    dtype=np.float64)
            param_values[:, :] = nominal_values
        self.param_values = param_values

        if initials is None:
            # Run simulation using same initial conditions
            species_names = [str(s) for s in self._model.species]
            initials = np.zeros(len(species_names))
            for ic in self._model.initial_conditions:
                initials[species_names.index(str(ic[0]))] = int(ic[1].value)
            initials = np.repeat([initials], param_values.shape[0], axis=0)
            self.initials = initials

        if tspan is None:
            tspan = self.tspan

        tout = [tspan] * len(param_values)
        t_out = np.array(tspan, dtype=np.float64)

        if threads is None:
            threads = self._threads

        size_params = param_values.shape[0]
        blocks, threads = self.get_blocks(size_params, threads)
        self._threads = threads
        self._blocks = blocks
        self._logger.info("Starting {} simulations on {} blocks"
                          "".format(number_sim, self._blocks))

        # compile kernel and send parameters to GPU
        if self._step_0:
            self._setup()

        timer_start = time.time()

        #  Note, this number will be larger than n_simulations if the gpu grid
        #  is not filled. The rest will be filled with zeros and not impact
        #  results. They are trimmed right before passing to simulation results

        # Not sure if this is needed for GPUs
        # total_threads = int(blocks * threads)
        # print(blocks, threads)
        total_threads = size_params
        # allocate and upload time to GPU

        time_points_gpu = array.to_device(
            self.queue,
            np.array(t_out, dtype=np.float64)
        )
        mem_order = 'C'
        param_array_gpu = array.to_device(
            self.queue,
            param_values.astype(np.float64).flatten(order=mem_order)
        )

        species_matrix_gpu = array.to_device(
            self.queue,
            initials.astype(np.int64).flatten(order=mem_order)
        )

        result_gpu = array.zeros(
            self.queue,
            order=mem_order,
            shape=(total_threads * len(t_out) * self._n_species,),
            dtype=np.int64
        )

        # perform simulation
        complete_event = self.program.Gillespie_all_steps(
            self.queue,
            (total_threads,),
            None,
            species_matrix_gpu.data,
            result_gpu.data,
            time_points_gpu.data,
            param_array_gpu.data,
            np.int64(len(t_out)),
            )
        complete_event.wait()
        # events = [complete_event]
        # Wait for kernel completion before host access
        # cl.enqueue_copy(self.queue, result_cpu, result_gpu, wait_for=events)

        self._time = time.time() - timer_start
        self._logger.info("{} simulations "
                          "in {}s".format(number_sim, self._time))

        # retrieve and store results, only keeping n_simulations
        # actual simulations we will return
        tout = np.array(tout)
        res = result_gpu.get(self.queue)
        res = res.reshape((total_threads, len(t_out), self._n_species))

        return SimulationResult(self, tout, res)

    def _setup(self):
        # print(self._code)
        with open('cl.cl', 'w') as f:
            f.write(self._code)
        self._compile(self._code)
        self._step_0 = False

    @staticmethod
    def _create_gpu_array(values, total_threads, prec):

        # Create species matrix on GPU
        # will make according to number of total threads, not n_simulations
        gpu_array = np.zeros((total_threads, values.shape[1]), dtype=prec)
        # Filling species matrix
        # Note that this might not fill entire array that was created.
        # The rest of the array will be zeros to fill up GPU.
        for i in range(len(values)):
            for j in range(values.shape[1]):
                gpu_array[i, j] = values[i, j]
        return gpu_array

    @staticmethod
    def get_blocks(n_simulations, threads_per_block):
        max_threads = 256
        if threads_per_block > max_threads:
            logging.warning("Limit of 256 threads per block due to curand."
                            " Setting to 256.")
            threads_per_block = max_threads
        if n_simulations < threads_per_block:
            block_count = 1
            threads_per_block = threads_per_block
        elif n_simulations % threads_per_block == 0:
            block_count = int(n_simulations // threads_per_block)
        else:
            block_count = int(n_simulations // threads_per_block + 1)
        return block_count, threads_per_block


def _lhs(model):
    """
    Left hand side
    """
    left_side = np.zeros((len(model.reactions), len(model.species)),
                         dtype=np.int32)
    for i in range(len(model.reactions)):
        for j in range(len(model.species)):
            stoich = 0
            for k in model.reactions[i]['reactants']:
                if j == k:
                    stoich += 1
            left_side[i, j] = stoich
    return left_side * -1


def _rhs(model):
    """

    Right hand side of matrix

    """
    right_side = np.zeros((len(model.reactions), len(model.species)),
                          dtype=np.int32)
    for i in range(len(model.reactions)):
        for j in range(len(model.species)):
            stoich = 0
            for k in model.reactions[i]['products']:
                if j == k:
                    stoich += 1
            right_side[i, j] = stoich
    return right_side


def _load_template():
    with open(os.path.join(os.path.dirname(__file__),
                           'pycuda_templates',
                           'opencl.cl'), 'r') as f:
        gillespie_code = f.read()
    return gillespie_code


if __name__ == '__main__':
    from pysb.examples.michment import model
    sim = GPUSimulatorCL(model)
    traj = sim.run(
        tspan=np.linspace(0, 20, 11),
        number_sim=10
    )

    result = traj.dataframe['Product']
    print(result.head(10))
    tout = result.index.levels[1].values
    result = result.unstack(0)
    result = result.as_matrix()
    import matplotlib.pyplot as plt

    plt.plot(tout, result, '0.5', lw=2, alpha=0.25)
    plt.show()
