from __future__ import print_function

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
        stoich_matrix = self._get_stoich(self.model)
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
            # Create expression strings with observables
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
            # replace param names with vector notation
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
        return dict(n_species=self._n_species, n_params=self._parameter_number,
                    n_reactions=_reaction_number, hazards=hazards_string,
                    stoch=stoich_string)

    def run(self, tspan=None, param_values=None, initials=None, number_sim=0):

        num_sim = int(number_sim)

        # check for proper arguments
        if param_values is None and initials is None and not num_sim:
            raise SimulatorException("Please provide a multi-dimension set of "
                                     "parameters, initials, or number_sim>0")
        elif param_values is None and not num_sim:
            num_sim = np.array(initials).shape[0]
        elif initials is None and not num_sim:
            num_sim = np.array(param_values).shape[0]

        if param_values is None and initials is None:
            # Run simulation using same param_values
            # initials will be taken care of on SimulatorBase side
            param_values = np.repeat(self.param_values, num_sim, axis=0)
        elif len(param_values.shape) == 1:
            # initials taken care of on SimulatorBase side
            param_values = np.repeat([param_values], num_sim, axis=0)
        elif len(initials.shape) == 1:
            # parameters taken care of on SimulatorBase side
            initials = np.repeat([initials], num_sim, axis=0)
        self.num_sim = num_sim
        super(SSABase, self).run(tspan=tspan, initials=initials,
                                 param_values=param_values,
                                 _run_kwargs=locals())

    @staticmethod
    def _get_stoich(model):
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
