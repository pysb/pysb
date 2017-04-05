from pysb.simulator.base import Simulator, SimulationResult, SimulatorException
from pysb.bng import generate_equations
from pysb.export.stochkit import StochKitExporter
import numpy as np
import subprocess
import os
import shutil
import tempfile
import re


class StochKitSimulator(Simulator):
    """
    Interface to the StochKit 2 stochastic simulation toolkit

    For more information on StochKit and to install it, please see its website
    https://engineering.ucsb.edu/~cse/StochKit/

    This class is inspired by the `gillespy
    <https://github.com/JohnAbel/gillespy>` library, but has been optimised
    for use with PySB.
    """
    _supports = {'multi_initials': True, 'multi_param_values': True}

    def __init__(self, model, tspan=None, cleanup=True, verbose=False):
        super(StochKitSimulator, self).__init__(model,
                                                tspan=tspan,
                                                verbose=verbose)
        self.cleanup = cleanup
        self._outdir = None
        generate_equations(self._model,
                           cleanup=self.cleanup,
                           verbose=self.verbose)

    def _run_stochkit(self, t=20, t_length=100, number_of_trajectories=1,
                      seed=None, algorithm='ssa', method=None,
                      num_processors=1, stats=False):

        extra_args = ''
        extra_args += '-p {:d}'.format(num_processors)

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

        # Find binary for selected algorithm (SSA, Tau-leaping, ...)
        if algorithm is None:
            raise SimulatorException("No StochKit algorithm selected")
        if not re.match(r'\w', algorithm):
            # Security check failure
            raise SimulatorException("StochKit algorithm name should contain "
                                     "only alphanumeric and underscore "
                                     "characters")
        executable = None
        if os.environ.get('STOCHKIT_HOME') is not None:
            if os.path.isfile(os.path.join(os.environ.get('STOCHKIT_HOME'),
                                           algorithm)):
                executable = os.path.join(os.environ.get('STOCHKIT_HOME'),
                                          algorithm)
        if executable is None:
            # try to find the executable in the path
            if os.environ.get('PATH') is not None:
                for dir in os.environ.get('PATH').split(':'):
                    if os.path.isfile(os.path.join(dir, algorithm)):
                        executable = os.path.join(dir, algorithm)
                        break
        if executable is None:
            raise SimulatorException("stochkit executable '{0}' not found. "
                                     "Make sure it is in your path, or set "
                                     "STOCHKIT_HOME environment "
                                     "variable.".format(algorithm))

        # Output model file to directory
        fname = os.path.join(self._outdir, 'pysb.xml')

        trajectories = []
        for i in range(len(self.initials)):
            # We write all StochKit output files to a temporary folder
            prefix_outdir = os.path.join(self._outdir, 'output_{}'.format(i))

            # Export model file
            with open(fname, 'w') as f:
                f.write(StochKitExporter(self._model).export(
                    self.initials[i], self.param_values[i]))

            # Assemble the argument list
            args = '--model {} --out-dir {} -t {:f} -i {:d}'.format(
                fname, prefix_outdir, t, t_length)

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
                raise SimulatorException("Solver execution failed: \
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
                '{0}' output: {1}{2}".format(cmd, stdout, stderr))

            traj_dir = os.path.join(prefix_outdir, 'trajectories')
            try:
                trajectories.extend([np.loadtxt(os.path.join(
                    traj_dir, f)) for f in os.listdir(traj_dir)])
            except Exception as e:
                raise SimulatorException(
                    "Error using solver.get_trajectories('{0}'): {1}".format(
                        prefix_outdir, e))

            if len(trajectories) == 0:
                raise SimulatorException("Solver execution failed: \
                '{0}' output: {1}{2}".format(cmd, stdout, stderr))

        if self.verbose:
            print("prefix_basedir={0}".format(self._outdir))
            print("STDOUT: {0}".format(stdout))
            if len(stderr) == 0:
                stderr = '<EMPTY>'
            print("STDERR: {0}".format(stderr))

        # Return data
        return trajectories

    def run(self, tspan=None, initials=None, param_values=None, n_runs=1,
            seed=None, output_dir=None, **additional_args):
        super(StochKitSimulator, self).run(tspan=tspan,
                                           initials=initials,
                                           param_values=param_values)

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
                                              **additional_args)
        finally:
            if self.cleanup:
                try:
                    shutil.rmtree(self._outdir)
                except OSError:
                    pass

        # set output time points
        self.tout = np.array(trajectories)[:, :, 0] + self.tspan[0]
        # species
        species = np.array(trajectories)[:, :, 1:]
        return SimulationResult(self, self.tout, species,
                                simulations_per_param_set=n_runs)
