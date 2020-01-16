from pysb.simulator.base import Simulator, SimulationResult
from typing import List
import pysb.bng

import numpy as np

from tempfile import mkdtemp

import sys
import os
import shutil
import importlib

try:
    import amici
except:
    amici = None


class AmiciSimulator(Simulator):
    """
    Simulate a model using amici (https://github.com/ICB-DCM/AMICI)

    Parameters
    ----------
    model : pysb.Model
        Model to simulate.
    tspan : vector-like, optional
        Time values over which to simulate. The first and last values define
        the time range. Returned trajectories are sampled at every value unless
        the simulation is interrupted for some reason, e.g., due to
        satisfaction of a logical stopping criterion (see 'tout' below).
    initials : vector-like or dict, optional
        Values to use for the initial condition of all species. Ordering is
        determined by the order of model.species. If not specified, initial
        conditions will be taken from model.initials (with initial condition
        parameter values taken from `param_values` if specified).
    param_values : vector-like or dict, optional
        Values to use for every parameter in the model. Ordering is
        determined by the order of model.parameters.
        If passed as a dictionary, keys must be parameter names.
        If not specified, parameter values will be taken directly from
        model.parameters.
    verbose : bool or int, optional (default: False)
        Sets the verbosity level of the logger. See the logging levels and
        constants from Python's logging module for interpretation of integer
        values. False is equal to the PySB default level (currently WARNING),
        True is equal to DEBUG.
    **kwargs : dict
        Extra keyword arguments, including:

        * ``modeldir``: directory in which the package with model code
        generated by amici will be stored. if valid, compiled model code
        already exist in that directory, the model will not be recompiled
        * ``force_recompile``: if this is set to true, the model will always be
        recompiled, independent of already existing model code

    """

    _supports = {'multi_initials': True,
                 'multi_param_values': True}

    def __init__(self, model, tspan=None, initials=None, param_values=None,
                 verbose=False, **kwargs):

        if amici is None:
            raise ImportError('AmiciSimulator requires a working installation'
                              'of amici. You can install amici via `pip '
                              'install amici`.')

        super(AmiciSimulator, self).__init__(model,
                                             tspan=tspan,
                                             initials=initials,
                                             param_values=param_values,
                                             verbose=verbose,
                                             **kwargs)

        self.modeldir_is_temp = 'modeldir' not in kwargs
        self.modeldir = kwargs.pop('modeldir',
                                    mkdtemp(prefix=f'pysbamici_{model.name}_'))
        force_recompile = kwargs.pop('force_recompile', False)

        if kwargs:
            raise ValueError('Unknown keyword argument(s): {}'.format(
                ', '.join(kwargs.keys())
            ))

        # Generate the equations for the model
        if force_recompile or not os.path.exists(os.path.join(self.modeldir,
                                                              model.name,
                                                              '__init__.py')):
            if not force_recompile and not self.modeldir_is_temp and \
                    os.path.exists(os.path.join(self.modeldir)):
                raise RuntimeError('Model directory already exists. Stopping '
                                   'to prevent data loss. To ignore this '
                                   'warning, pass `force_recompile=True`')

            amici.pysb2amici(model,
                             self.modeldir,
                             verbose=False,
                             observables=[],
                             constant_parameters=[],
                             compute_conservation_laws=True)
            mode = 'compilation'
            help = 'file an issue at https://github.com/ICB-DCM/AMICI/issues.'
        else:
            pysb.bng.generate_equations(model)
            mode = 'loading'
            help = 'try recompiling the model by passing ' \
                   '`force_recompile=True`.'

        # Load the generated model package
        sys.path.insert(0, self.modeldir)
        try:
            modelModulePYSB = importlib.import_module(model.name)
        except Exception as e:
            raise RuntimeError(f'Model {mode} failed. Please {help}')

        self._model = model
        self.amici_model = modelModulePYSB.getModel()
        self.amici_solver = self.amici_model.getSolver()
        self.amici_model.setTimepoints(tspan if tspan is not None else [])

    def __del__(self):
        # if we generated a temporary directory using mkdtemp, we are
        # responsible of cleaning up the directory afterwards
        if self.modeldir_is_temp:
            shutil.rmtree(self.modeldir)

    def run(self, tspan=None, initials=None, param_values=None,
            num_processors=1):
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
        num_processors : int
            Number of processes to use (default: 1). Set to a larger number
            (e.g. the number of CPU cores available) for parallel execution of
            simulations. This is only useful when simulating with more than one
            set of initial conditions and/or parameters.

        Returns
        -------
        A :class:`SimulationResult` object
        """
        super(AmiciSimulator, self).run(tspan=tspan,
                                        initials=initials,
                                        param_values=param_values,
                                        _run_kwargs=[])
        n_sims = len(self.param_values)

        num_processors = min(n_sims, num_processors)

        if num_processors == 1:
            self._logger.debug('Single processor (serial) mode')
        else:
            if not amici.compiledWithOpenMP():
                raise EnvironmentError(
                    'AMICI/model was not compiled with openMP support, which '
                    'is required for parallel simulation. Please see '
                    'https://github.com/ICB-DCM/AMICI/blob/master'
                    '/documentation/PYTHON.md for details on how to compile '
                    'AMICI with openMP support.')
            self._logger.debug('Multi-processor (parallel) mode using {} '
                               'processes'.format(num_processors))

        edatas = self._simulationspecs_to_edatas()

        rdatas = amici.runAmiciSimulations(
            model=self.amici_model, solver=self.amici_solver,
            edata_list=edatas, failfast=False, num_threads=num_processors
        )

        self._logger.info('All simulation(s) complete')
        return SimulationResult(self, np.array([self.tspan] * n_sims),
                                self._rdatas_to_trajectories(rdatas))

    def _simulationspecs_to_edatas(self) -> List:
        """ Converts tspan, param_values and initials into amici.ExpData
        objects """
        n_sims = len(self.param_values)

        edatas = [
            amici.ExpData(self.amici_model.get())
            for _ in range(n_sims)
        ]

        for isim, edata in enumerate(edatas):
            edata.setTimepoints(self.tspan)
            edata.parameters = self._pysb2amici_parameters(
                self.param_values[isim]
            )
            edata.fixedParameters = self._pysb2amici_fixed_parameters(
                self.param_values[isim]
            )
            edata.x0 = self._pysb2amici_initials(
                self.initials[max(isim, len(self.initials) - 1)]
            )

        return edatas

    def _pysb2amici_parameters(self, parameters: List[float]):
        """ Reorders and maps pysb parameters to amici parameters """
        return [
            parameters[self.model.parameters.keys().index(amici_par_name)]
            for amici_par_name in self.amici_model.getParameterIds()
        ]

    def _pysb2amici_fixed_parameters(self, parameters: List[float]):
        """ Reorders and maps pysb parameters to amici constants """
        return [
            parameters[self.model.parameters.keys().index(amici_par_name)]
            for amici_par_name in self.amici_model.getFixedParameterIds()
        ]

    def _pysb2amici_initials(self, initials: List[float]):
        """ Reorders and maps pysb species to amici states variables """
        states = [f'__s{ix}' for ix in range(len(self.model.species))]
        return [
            initials[states.index(amici_par_name)]
            for amici_par_name in self.amici_model.getStateIds()
        ]

    def _rdatas_to_trajectories(self, rdatas: List) -> List:
        """ Extracts state trajectories from lists of amici.ReturnData  """
        return [
            np.asarray(rdata['x'])
            for rdata in rdatas
        ]
