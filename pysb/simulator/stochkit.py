from pysb.simulator.base import Simulator, SimulationResult, SimulatorException
from pysb.bng import generate_equations
from pysb.export.stochkit import StochKitExporter
import numpy as np
import subprocess
import os
import shutil
import tempfile
from pysb.pathfinder import get_path
from pysb.logging import EXTENDED_DEBUG


class StochKitSimulator(Simulator):
    """
    Interface to the StochKit 2 stochastic simulation toolkit

    StochKit can be installed from GitHub:
    https://github.com/stochss/stochkit

    This class is inspired by the `gillespy
    <https://github.com/JohnAbel/gillespy>` library, but has been optimised
    for use with PySB.

    .. warning::
        The interface for this class is considered experimental and may
        change without warning as PySB is updated.

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
        conditions will be taken from model.initial_conditions (with
        initial condition parameter values taken from `param_values` if
        specified).
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

        * ``cleanup``: Boolean, delete directory after completion if True

    Examples
    --------
    Simulate a model and display the results for an observable:

    >>> from pysb.examples.robertson import model
    >>> import numpy as np
    >>> np.set_printoptions(precision=4)
    >>> sim = StochKitSimulator(model, tspan=np.linspace(0, 10, 5))

    Here we supply a "seed" to the random number generator for deterministic
    results, but for most purposes it is recommended to leave this blank.

    >>> simulation_result = sim.run(n_runs=2, seed=123456)

    A_total trajectory for first run

    >>> print(simulation_result.observables[0]['A_total']) \
        #doctest: +NORMALIZE_WHITESPACE
    [1.  0.  0.  0.  0.]

    A_total trajectory for second run

    >>> print(simulation_result.observables[1]['A_total']) \
        #doctest: +SKIP
    [1.  1.  1.  0.  0.]

    For further information on retrieving trajectories (species,
    observables, expressions over time) from the ``simulation_result``
    object returned by :func:`run`, see the examples under the
    :class:`SimulationResult` class.
    """
    _supports = {'multi_initials': True, 'multi_param_values': True}

    def __init__(self, model, tspan=None, initials=None,
                 param_values=None, verbose=False, **kwargs):
        super(StochKitSimulator, self).__init__(model,
                                                tspan=tspan,
                                                initials=initials,
                                                param_values=param_values,
                                                verbose=verbose,
                                                **kwargs)
        self.cleanup = kwargs.pop('cleanup', True)
        if kwargs:
            raise ValueError('Unknown keyword argument(s): {}'.format(
                ', '.join(kwargs.keys())
            ))
        self._outdir = None
        generate_equations(self._model,
                           cleanup=self.cleanup,
                           verbose=self.verbose)

    def _run_stochkit(self, t=20, t_length=100, number_of_trajectories=1,
                      seed=None, algorithm='ssa', method=None,
                      num_processors=1, stats=False, epsilon=None,
                      threshold=None):

        extra_args = '-p {:d}'.format(num_processors)

        # Random seed for stochastic simulation
        if seed is not None:
            extra_args += ' --seed {:d}'.format(seed)

        # Keep all the trajectories by default
        extra_args += ' --keep-trajectories'

        # Number of trajectories
        extra_args += ' --realizations {:d}'.format(number_of_trajectories)

        # We generally don't need the extra stats
        if not stats:
            extra_args += ' --no-stats'

        if method is not None:  # This only works for StochKit 2.1
            extra_args += ' --method {}'.format(method)

        if epsilon is not None:
            extra_args += ' --epsilon {:f}'.format(epsilon)

        if threshold is not None:
            extra_args += ' --threshold {:d}'.format(threshold)

        # Find binary for selected algorithm (SSA, Tau-leaping, ...)
        if algorithm not in ['ssa', 'tau_leaping']:
            raise SimulatorException(
                "algorithm must be 'ssa' or 'tau_leaping'")

        executable = get_path('stochkit_{}'.format(algorithm))

        # Output model file to directory
        fname = os.path.join(self._outdir, 'pysb.xml')

        trajectories = []
        for i in range(len(self.initials)):
            # We write all StochKit output files to a temporary folder
            prefix_outdir = os.path.join(self._outdir, 'output_{}'.format(i))

            # Export model file
            stoch_xml = StochKitExporter(self._model).export(
                self.initials[i], self.param_values[i])
            self._logger.log(EXTENDED_DEBUG, 'StochKit XML:\n%s' % stoch_xml)
            with open(fname, 'wt') as f:
                f.write(stoch_xml)

            # Assemble the argument list
            args = '--model {} --out-dir {} -t {:f} -i {:d}'.format(
                fname, prefix_outdir, t, t_length - 1)

            # If we are using local mode, shell out and run StochKit
            # (SSA or Tau-leaping or ODE)
            cmd = '{} {} {}'.format(executable, args, extra_args)
            self._logger.debug("StochKit run {} of {} (cmd: {})".format(
                (i + 1), len(self.initials), cmd))

            # Execute
            try:
                handle = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, shell=True)
                return_code = handle.wait()
            except OSError as e:
                raise SimulatorException("StochKit execution failed: \
                {0}\n{1}".format(cmd, e))

            try:
                stderr = handle.stderr.read()
            except Exception as e:
                stderr = 'Error reading stderr: {0}'.format(e)
            try:
                stdout = handle.stdout.read()
            except Exception as e:
                stdout = 'Error reading stdout: {0}'.format(e)

            if return_code != 0:
                raise SimulatorException("Solver execution failed: \
                '{0}' output:\nSTDOUT:\n{1}\nSTDERR:\n{2}".format(
                    cmd, stdout, stderr))

            traj_dir = os.path.join(prefix_outdir, 'trajectories')
            try:
                trajectories.extend([np.loadtxt(os.path.join(
                    traj_dir, f)) for f in sorted(os.listdir(traj_dir))])
            except Exception as e:
                raise SimulatorException(
                    "Error reading StochKit trajectories: {0}"
                    "\nSTDOUT:{1}\nSTDERR:{2}".format(e, stdout, stderr))

            if len(trajectories) == 0 or len(stderr) != 0:
                raise SimulatorException("Solver execution failed: \
                '{0}' output:\nSTDOUT:\n{1}\nSTDERR:\n{2}".format(
                    cmd, stdout, stderr))

            self._logger.debug("StochKit STDOUT:\n{0}".format(stdout))

        # Return data
        return trajectories

    def run(self, tspan=None, initials=None, param_values=None, n_runs=1,
            algorithm='ssa', output_dir=None,
            num_processors=1, seed=None, method=None, stats=False,
            epsilon=None, threshold=None):
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
            See parameter definitions in :class:`StochKitSimulator`.
        n_runs : int
            The number of simulation runs per parameter set. The total
            number of simulations is therefore n_runs * max(len(initials),
            len(param_values))
        algorithm : str
            Choice of 'ssa' (Gillespie's stochastic simulation algorithm) or
            'tau_leaping' (Tau leaping algorithm)
        output_dir : str or None
            Directory for StochKit output, or None for a system-specific
            temporary directory
        num_processors : int
            Number of CPU cores for StochKit to use (default: 1)
        seed : int or None
            A random number seed for StochKit. Set to any integer value for
            deterministic behavior.
        method : str or None
            StochKit "method" argument, default None. Only used by StochKit
            2.1 (not yet released at time of writing).
        stats : bool
            Ask StochKit to generate simulation summary statistics if True
        epsilon : float or None
            Tolerance parameter for tau-leaping algorithm
        threshold : int or None
            Threshold parameter for tau-leaping algorithm

        Returns
        -------
        A :class:`SimulationResult` object
        """
        super(StochKitSimulator, self).run(tspan=tspan,
                                           initials=initials,
                                           param_values=param_values,
                                           _run_kwargs=locals())

        self._logger.info('Running StochKit with {:d} parameter sets, '
                          '{:d} repeats ({:d} simulations total)'.format(
                           len(self.initials), n_runs, len(self.initials) *
                           n_runs))

        if output_dir is None:
            self._outdir = tempfile.mkdtemp()
        else:
            self._outdir = output_dir

        # Calculate time intervals and validate
        t_range = self.tspan[-1] - self.tspan[0]
        t_length = len(self.tspan)
        if not np.allclose(self.tspan, np.linspace(0, self.tspan[-1],
                                                   t_length)):
            raise SimulatorException('StochKit requires tspan to be linearly '
                                     'spaced starting at t=0')

        try:
            trajectories = self._run_stochkit(t=t_range,
                                              number_of_trajectories=n_runs,
                                              t_length=t_length,
                                              seed=seed,
                                              algorithm=algorithm,
                                              method=method,
                                              num_processors=num_processors,
                                              stats=stats,
                                              epsilon=epsilon,
                                              threshold=threshold)
        finally:
            if self.cleanup:
                try:
                    shutil.rmtree(self._outdir)
                except OSError:
                    pass

        # set output time points
        trajectories_array = np.array(trajectories)
        self.tout = trajectories_array[:, :, 0] + self.tspan[0]
        # species
        species = trajectories_array[:, :, 1:]
        return SimulationResult(self, self.tout, species,
                                simulations_per_param_set=n_runs)
