import logging
import re

import numpy as np
import sympy

from pysb.bng import generate_equations
from pysb.core import Expression
from pysb.logging import setup_logger
from pysb.simulator.base import Simulator, SimulatorException


class SSABase(Simulator):
    _supports = {'multi_initials': True, 'multi_param_values': True}

    def __init__(self, model, verbose=False, tspan=None, **kwargs):
        super(SSABase, self).__init__(model, verbose, **kwargs)

        generate_equations(self._model)

        self.tspan = tspan
        self.verbose = verbose

        # private attribute
        self._parameter_number = len(self._model.parameters)
        self._n_species = len(self._model.species)
        self._n_reactions = len(self._model.reactions)
        self._step_0 = True
        self.num_sim = None
        if verbose:
            setup_logger(logging.INFO)

    def _get_template_args(self):
        """ converts pysb reactions to pycuda/pyopencl format """
        p = re.compile('\s')
        stoich_matrix = self.model.stoichiometry_matrix.toarray()
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
        output_string = ''

        expr_strings = {
            e.name: '(%s)' % sympy.ccode(
                e.expand_expr(expand_observables=True)
            ) for e in self.model.expressions}
        for n, rxn in enumerate(self._model.reactions):
            output_string += "\th[%s] = " % repr(n)
            rate = sympy.fcode(rxn["rate"])
            rate = re.sub('d0', '', rate)
            rate = p.sub('', rate)

            # Create expression strings with observables
            # expand only expressions used in the rate eqn
            for e in {sym for sym in rxn["rate"].atoms()
                      if isinstance(sym, Expression)}:
                rate = re.sub(r'\b%s\b' % e.name,
                              expr_strings[e.name],
                              rate)

            # replace x**2 with (x-1)*x
            pattern = "(_{2}s\d+)\*\*(\d+)"
            matches = re.findall(pattern, rate)
            for m in matches:
                repl = m[0]
                for i in range(1, int(m[1])):
                    repl += "*(%s-%d)" % (m[0], i)
                rate = re.sub(pattern, repl, rate)
            # replace species string with matrix index (`_si` with `y[i]`)
            rate = re.sub(r'_{2}s(\d+)', lambda m: 'y[%s]' % (int(m.group(1))),
                          rate)
            # replace param names with vector notation
            for q, prm in enumerate(params_names):
                rate = re.sub(r'\b(%s)\b' % prm, 'param_vec[%s]' % q, rate)

            # Calculate the fast approximate, better performance on GPUs
            rate = rate.replace('pow', 'powf')
            # If a parameter is a float and appears first, the result output
            # will lose precision. Casting to double ensures precision
            rate = '(double)' + rate

            output_string += rate + ";\n"
        return dict(n_species=self._n_species, n_params=self._parameter_number,
                    n_reactions=_reaction_number, propensities=output_string,
                    stoch=stoich_string)

    def run(self, tspan=None, param_values=None, initials=None, number_sim=0):

        num_sim = int(number_sim)

        # check for proper arguments
        if param_values is None and initials is None and not num_sim:
            raise SimulatorException("Please provide a multi-dimension set of "
                                     "parameters, initials, or number_sim>0")
        elif param_values is None and not num_sim:
            self.initials = initials
            num_sim = self.initials.shape[0]
        elif initials is None and not num_sim:
            self.param_values = param_values
            num_sim = self.param_values.shape[0]

        if param_values is None and initials is None:
            # Run simulation using same param_values
            # initials will be taken care of on SimulatorBase side
            param_values = np.repeat(self.param_values, num_sim, axis=0)
        elif isinstance(param_values, np.ndarray) and len(
                param_values.shape) == 1:
            # initials taken care of on SimulatorBase side
            param_values = np.repeat([param_values], num_sim, axis=0)
        elif isinstance(initials, np.ndarray) and len(initials.shape) == 1:
            # parameters taken care of on SimulatorBase side
            initials = np.repeat([initials], num_sim, axis=0)
        self.num_sim = num_sim
        super(SSABase, self).run(tspan=tspan, initials=initials,
                                 param_values=param_values,
                                 _run_kwargs=locals())

    @staticmethod
    def get_blocks(n_simulations, threads_per_block):
        # Choosing the number of blocks and threads per block depends on the
        # hardware warpsize (32 for NVIDIA, 64 for AMD), and number of
        # simulations.
        # CUDA CURAND limits it to 256, so that's the hard limit we set.
        # We want to number of blocks to be a multiple of the
        # threads_per_block, so we saturate the GPU equally.
        # If the number of simulations isn't a multiple of it,
        # we just make the number of blocks slightly bigger, then fill
        # the rest of the space with zeros, which instantly finishes.

        max_tpb = 256
        if threads_per_block > max_tpb:
            # Limit of 256 threads per block from curand
            threads_per_block = max_tpb

        if n_simulations < max_tpb:
            block_count = 1
            threads_per_block = max_tpb
        elif n_simulations % threads_per_block == 0:
            block_count = int(n_simulations // threads_per_block)
        else:
            block_count = int(n_simulations // threads_per_block + 1)
        return block_count, threads_per_block
