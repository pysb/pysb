from pysb.simulator.base import Simulator, SimulationResult, SimulatorException
from pysb.bng import BngFileInterface, load_equations, generate_hybrid_model
import numpy as np
import logging
from pysb.logging import EXTENDED_DEBUG
from pysb.core import as_complex_pattern, Parameter, \
    InvalidComplexPatternException
import collections
import os


class BngSimulator(Simulator):
    _supports = {
        'multi_initials':     True,
        'multi_param_values': True
    }
    _SIMULATOR_TYPES = ['ssa', 'nf', 'pla', 'ode']

    def __init__(self, model, tspan=None, cleanup=True, verbose=False):
        super(BngSimulator, self).__init__(model, tspan=tspan,
                                           verbose=verbose)
        self.cleanup = cleanup
        self._outdir = None

    def run(self, tspan=None, initials=None, param_values=None, n_runs=1,
            method='ssa', output_dir=None, output_file_basename=None,
            cleanup=True, population_maps=None, **additional_args):
        """
        Simulate a model using BioNetGen

        Parameters
        ----------
        tspan: vector-like
            time span of simulation
        initials: vector-like, optional
            initial conditions of model
        param_values : vector-like or dictionary, optional
            Values to use for every parameter in the model. Ordering is
            determined by the order of model.parameters.
            If not specified, parameter values will be taken directly from
            model.parameters.
        n_runs: int
            number of simulations to run
        method : str
            Type of simulation to run. Must be one of:

             * 'ssa' - Stochastic Simulation Algorithm (direct method with
             propensity sorting)
             * 'nf' - Stochastic network free simulation with NFsim.
             Performs Hybrid Particle/Population simulation if population_maps
             argument is supplied
             * 'pla' - Partioned-leaping algorithm (variant of tau-leaping
             algorithm)
             * 'ode' - ODE simulation (Sundials CVODE algorithm)

        output_dir : string, optional
            Location for temporary files generated by BNG. If None (the
            default), uses a temporary directory provided by the system. A
            temporary directory with a random name is created within the
            supplied location.
        output_file_basename : string, optional
            This argument is used as a prefix for the temporary BNG
            output directory, rather than the individual files.
        cleanup : bool, optional
            If True (default), delete the temporary files after the
            simulation is
            finished. If False, leave them in place. Useful for debugging.
        population_maps: list of PopulationMap
            List of :py:class:`PopulationMap` objects for hybrid
            particle/population modeling. Only used when method='nf'.
        additional_args: kwargs, optional
            Additional arguments to pass to BioNetGen

        """
        super(BngSimulator, self).run(tspan=tspan,
                                      initials=initials,
                                      param_values=param_values
                                      )

        if method not in self._SIMULATOR_TYPES:
            raise ValueError("Method must be one of " +
                             str(self._SIMULATOR_TYPES))

        if method != 'nf' and population_maps:
            raise ValueError('population_maps argument is only used when '
                             'method is "nf"')

        if method == 'nf':
            if population_maps is not None and (not isinstance(
                    population_maps, collections.Iterable) or
                    any(not isinstance(pm, PopulationMap) for pm in
                        population_maps)):
                raise ValueError('population_maps should be a list of '
                                 'PopulationMap objects')

            if not np.allclose(self.tspan, np.linspace(0, self.tspan[-1],
                                                       len(self.tspan))):
                raise SimulatorException('NFsim requires tspan to be linearly '
                                         'spaced starting at t=0')
            additional_args['t_end'] = self.tspan[-1]
            additional_args['n_steps'] = len(self.tspan) - 1
            model_additional_species = self.model.species
        else:
            additional_args['sample_times'] = self.tspan
            model_additional_species = None

        additional_args['method'] = method
        additional_args['print_functions'] = True
        verbose_bool = self._logger.logger.getEffectiveLevel() <= logging.DEBUG
        extended_debug = self._logger.logger.getEffectiveLevel() <= \
                         EXTENDED_DEBUG
        additional_args['verbose'] = extended_debug
        params_names = [g.name for g in self._model.parameters]

        n_param_sets = self.initials_length
        total_sims = n_runs * n_param_sets

        self._logger.info('Running %d BNG %s simulations' % (total_sims,
                                                             method))

        model_to_load = None
        hpp_bngl = None

        if population_maps:
            self._logger.debug('Generating hybrid particle-population model')
            hpp_bngl = generate_hybrid_model(
                self._model,
                population_maps,
                model_additional_species,
                verbose=extended_debug)
        else:
            model_to_load = self._model

        with BngFileInterface(model_to_load,
                              verbose=verbose_bool,
                              output_dir=output_dir,
                              output_prefix=output_file_basename,
                              cleanup=cleanup,
                              model_additional_species=model_additional_species
                              ) as bngfile:
            if hpp_bngl:
                hpp_bngl_filename = os.path.join(bngfile.base_directory,
                                                 'hpp_model.bngl')
                self._logger.debug('HPP BNGL:\n\n' + hpp_bngl)
                with open(hpp_bngl_filename, 'w') as f:
                    f.write(hpp_bngl)
            if method != 'nf':
                # TODO: Write existing netfile if already generated
                bngfile.action('generate_network', overwrite=True,
                               verbose=extended_debug)
            if output_file_basename is None:
                prefix = 'pysb'
            else:
                prefix = output_file_basename

            sim_prefix = 0
            for pset_idx in range(n_param_sets):
                for n in range(len(self.param_values[pset_idx])):
                    bngfile.set_parameter(params_names[n],
                                          self.param_values[pset_idx][n])
                for cp, values in self.initials_dict.items():
                    if population_maps:
                        for pm in population_maps:
                            if pm.complex_pattern.is_equivalent_to(cp):
                                cp = pm.counter_species
                                break
                    bngfile.set_concentration(cp, values[pset_idx])
                for sim_rpt in range(n_runs):
                    tmp = additional_args.copy()
                    tmp['prefix'] = '{}{}'.format(prefix, sim_prefix)
                    bngfile.action('simulate', **tmp)
                    bngfile.action('resetConcentrations')
                    sim_prefix += 1
            if hpp_bngl:
                bngfile.execute(reload_netfile=hpp_bngl_filename,
                                skip_file_actions=True)
            else:
                bngfile.execute()
            if method != 'nf':
                load_equations(self.model, bngfile.net_filename)
            list_of_yfull = \
                BngFileInterface.read_simulation_results_multi(
                [bngfile.base_filename + str(n) for n in range(total_sims)])

        tout = []
        species_out = []
        obs_exp_out = []
        for i in range(total_sims):
            yfull = list_of_yfull[i]
            yfull_view = yfull.view(float).reshape(len(yfull), -1)

            tout.append(yfull_view[:, 0])

            if method == 'nf':
                obs_exp_out.append(yfull_view[:, 1:])
            else:
                species_out.append(yfull_view[:,
                                   1:(len(self.model.species) + 1)])
                if len(self.model.observables) or len(self.model.expressions):
                    obs_exp_out.append(yfull_view[:,
                                       (len(self.model.species) + 1):])

        return SimulationResult(self, tout=tout, trajectories=species_out,
                                observables_and_expressions=obs_exp_out,
                                simulations_per_param_set=n_runs)


class PopulationMap(object):
    """
    Population map for BioNetGen hybrid particle/population simulation

    References
    ----------

    Hogg et al. 2014:
    http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003544

    BioNetGen HPP documentation:
    http://bionetgen.org/index.php/Hybrid_particle-population_model_generator

    """
    def __init__(self, complex_pattern, lumping_rate, counter_species=None):
        try:
            self.complex_pattern = as_complex_pattern(complex_pattern)
        except InvalidComplexPatternException:
            raise ValueError('complex_pattern must be a ComplexPattern')

        if not isinstance(lumping_rate, Parameter):
            raise ValueError('lumping_rate must be a %s' % Parameter.__class__)

        self.lumping_rate = lumping_rate
        if counter_species is None:
            self.counter_species = None
        else:
            self.counter_species = str(counter_species)

    def __repr__(self):
        return 'PopulationMap({}, {}, {})'.format(
            self.complex_pattern,
            self.lumping_rate,
            'None' if self.counter_species is None else '{}'.format(
                self.counter_species)
        )
