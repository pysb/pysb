from __future__ import absolute_import
from __future__ import print_function as _
import pysb.core
from pysb.generator.bng import BngGenerator, format_complexpattern
import os
import subprocess
import re
import itertools
import sympy
import numpy
import tempfile
import abc
from warnings import warn
import shutil
import collections
import pysb.pathfinder as pf
from pysb.logging import get_logger, EXTENDED_DEBUG
import logging

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
try:
    from future_builtins import zip
except ImportError:
    pass

# Alias basestring under Python 3 for forwards compatibility
try:
    basestring
except NameError:
    basestring = str


def set_bng_path(dir):
    """ Deprecated. Use pysb.pathfinder.set_path() instead. """
    warn("Function %s() is deprecated; use pysb.pathfinder.set_path() "
         "instead" % set_bng_path.__name__, category=DeprecationWarning,
         stacklevel=2)
    pf.set_path('bng', dir)


class BngInterfaceError(RuntimeError):
    """BNG reported an error"""
    pass


class BngBaseInterface(object):
    """ Abstract base class for interfacing with BNG """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, model=None, verbose=False, cleanup=False,
                 output_prefix=None, output_dir=None,
                 model_additional_species=None,
                 model_population_maps=None):
        self._logger = get_logger(__name__,
                                  model=model,
                                  log_level=verbose)
        self._base_file_stem = 'pysb'
        self.cleanup = cleanup
        self.output_prefix = 'tmpBNG' if output_prefix is None else \
            output_prefix
        if model:
            self.generator = BngGenerator(
                model, additional_initials=model_additional_species,
                population_maps=model_population_maps
            )
            self.model = self.generator.model
            self._check_model()
        else:
            self.generator = None
            self.model = None

        self.base_directory = tempfile.mkdtemp(prefix=self.output_prefix,
                                               dir=output_dir)
        self._logger.debug('{} instantiated in directory {}'.format(
            self.__class__, self.base_directory)
        )

    def __enter__(self):
        return self

    @abc.abstractmethod
    def __exit__(self):
        return

    def _delete_tmpdir(self):
        shutil.rmtree(self.base_directory)

    def _check_model(self):
        """
        Checks a model has at least one initial condition and rule, raising
        an exception if not
        """
        if not self.model.rules:
            raise NoRulesError()
        if not self.model.initial_conditions and not any(r.is_synth() for r
                                                         in self.model.rules):
            raise NoInitialConditionsError()

    @classmethod
    def _bng_param(cls, param):
        """
        Ensures a BNG console parameter is in the correct format

        Strings are double quoted and booleans are mapped to [0,1]. Other
        types are currently used verbatim.

        Parameters
        ----------
        param :
            An argument to a BNG action call
        """
        if isinstance(param, basestring):
            return '"%s"' % param
        elif isinstance(param, bool):
            return 1 if param else 0
        elif isinstance(param, (collections.Sequence, numpy.ndarray)):
            return list(param)
        return param

    @abc.abstractmethod
    def action(self, action, **kwargs):
        """
        Generates code to execute a BNG action command

        Parameters
        ----------
        action: string
            The name of the BNG action function
        kwargs: kwargs, optional
            Arguments and values to supply to BNG
        """
        return

    @classmethod
    def _format_action_args(cls, **kwargs):
        """
        Formats a set of arguments for BNG

        Parameters
        ----------
        kwargs: kwargs, optional
            Arguments and values to supply to BNG
        """
        if kwargs:
            action_args = ','.join('%s=>%s' % (k, BngConsole._bng_param(v))
                                   for k, v in kwargs.items())
        else:
            action_args = ''
        return action_args

    @property
    def base_filename(self):
        """
        Returns the base filename (without extension) for BNG output files
        """
        return os.path.join(self.base_directory, self._base_file_stem)

    @property
    def bng_filename(self):
        """
        Returns the BNG command list (.bngl) filename (does not check
        whether the file exists)
        """
        return self.base_filename + '.bngl'

    @property
    def net_filename(self):
        """
        Returns the BNG network filename (does not check whether the file
        exists)
        """
        return self.base_filename + '.net'

    def read_netfile(self):
        """
        Reads a BNG network file as a string. Note that you must execute
        network generation separately before attempting this, or the file will
        not be found.
        :return: Contents of the BNG network file as a string
        """
        self._logger.debug('Reading BNG netfile: %s' % self.net_filename)
        with open(self.net_filename, 'r') as net_file:
            output = net_file.read()

        return output

    def read_simulation_results(self):
        """
        Read the results of a BNG simulation as a numpy array

        Returns
        -------
        numpy.ndarray
            Simulation results in a 2D matrix (time on Y axis,
            species/observables/expressions on X axis depending on
            simulation type)
        """
        return self.read_simulation_results_multi([self.base_filename])[0]

    @staticmethod
    def read_simulation_results_multi(base_filenames):
        """
        Read the results of multiple BNG simulations

        Parameters
        ----------
        base_filenames: list of str
            A list of filename stems to read simulation results in from,
            including the full path but not including any file extension.

        Returns
        -------
        list of numpy.ndarray
            List of simulation results, each in a 2D matrix (time on Y axis,
            species/observables/expressions on X axis depending on
            simulation type)
        """
        list_of_yfulls = []
        for base_filename in base_filenames:
            names = ['time']

            # Read concentrations data
            try:
                cdat_arr = numpy.loadtxt(base_filename + '.cdat', skiprows=1)
                # -1 for time column
                names += ['__s%d' % i for i in range(cdat_arr.shape[1] - 1)]
            except IOError:
                cdat_arr = None

            # Read groups data
            try:
                with open(base_filename + '.gdat', 'r') as f:
                    # Exclude \# and time column
                    names += f.readline().split()[2:]
                    # Exclude first column (time)
                    gdat_arr = numpy.loadtxt(f)
                    if cdat_arr is None:
                        cdat_arr = numpy.ndarray((len(gdat_arr), 0))
                    else:
                        gdat_arr = gdat_arr[:, 1:]
            except IOError:
                if cdat_arr is None:
                    raise BngInterfaceError('Need at least one of .cdat file or '
                                            '.gdat file to read simulation '
                                            'results')
                gdat_arr = numpy.ndarray((len(cdat_arr), 0))

            yfull_dtype = list(zip(names, itertools.repeat(float)))
            yfull = numpy.ndarray(len(cdat_arr), yfull_dtype)

            yfull_view = yfull.view(float).reshape(len(yfull), -1)
            yfull_view[:, :cdat_arr.shape[1]] = cdat_arr
            yfull_view[:, cdat_arr.shape[1]:] = gdat_arr

            list_of_yfulls.append(yfull)

        return list_of_yfulls


class BngConsole(BngBaseInterface):
    """ Interact with BioNetGen through BNG Console """
    def __init__(self, model=None, verbose=False, cleanup=True,
                 output_dir=None, output_prefix=None, timeout=30,
                 suppress_warnings=False, model_additional_species=None):
        super(BngConsole, self).__init__(
            model, verbose, cleanup, output_prefix, output_dir,
            model_additional_species=model_additional_species
        )

        try:
            import pexpect
        except ImportError:
            raise ImportError("Library 'pexpect' is required to use "
                              "BNGConsole, please install it to continue.\n"
                              "It is not currently available on Windows.")

        if suppress_warnings:
            warn("suppress_warnings is deprecated and has no effect. Adjust "
                 "the log level with the verbose argument instead.",
                 category=DeprecationWarning,
                 stacklevel=2)

        # Generate BNGL file
        if self.model:
            with open(self.bng_filename, mode='w') as bng_file:
                bng_file.write(self.generator.get_content())

        # Start BNG Console and load BNGL
        bng_path = pf.get_path('bng')
        bng_exec_path = '%s --console' % bng_path
        if not bng_path.endswith('.bat'):
            bng_exec_path = 'perl %s' % bng_exec_path
        self.console = pexpect.spawn(bng_exec_path,
                                     cwd=self.base_directory,
                                     timeout=timeout)
        self._console_wait()
        if self.model:
            self.console.sendline('load %s' % self.bng_filename)
            self._console_wait()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        In console mode, commands have already been executed, so we simply
        close down the console and erase the temporary directory if applicable.
        """
        self.console.sendline('done')
        self.console.close()
        if self.cleanup:
            self._delete_tmpdir()

    def _console_wait(self):
        """
        Wait for BNG console to process the command, and return the output
        :return: BNG console output from the previous command
        """
        self.console.expect('BNG>')
        # Python 3 requires explicit conversion of 'bytes' to 'str'
        console_msg = self.console.before.decode('utf-8')
        if "ERROR:" in console_msg:
            raise BngInterfaceError(console_msg)
        elif "WARNING:" in console_msg:
            self._logger.warning(console_msg)
        else:
            self._logger.debug(console_msg)
        return console_msg

    def generate_network(self, overwrite=False):
        """
        Generates a network in BNG and returns the network file contents as
        a string

        Parameters
        ----------
        overwrite: bool, optional
            Overwrite existing network file, if any
        """
        self.action('generate_network', overwrite=overwrite)
        return self.read_netfile()

    def action(self, action, **kwargs):
        """
        Generates a BNG action command and executes it through the console,
        returning any console output

        Parameters
        ----------
        action : string
            The name of the BNG action function
        kwargs : kwargs, optional
            Arguments and values to supply to BNG
        """
        # Process BNG arguments into a string
        action_args = self._format_action_args(**kwargs)

        # Execute the command via the console
        if action_args == '':
            cmd = 'action %s()' % action
        else:
            cmd = 'action %s({%s})' % (action, action_args)
        self._logger.debug(cmd)
        self.console.sendline(cmd)

        # Wait for the command to execute and return the result
        return self._console_wait()

    def load_bngl(self, bngl_file):
        """
        Load a BNGL file in the BNG console

        Parameters
        ----------
        bngl_file : string
            The filename of a .bngl file
        """
        cmd = 'load %s' % bngl_file
        self._logger.debug(cmd)
        self.console.sendline(cmd)
        self._console_wait()
        self._base_file_stem = os.path.splitext(os.path.basename(bngl_file))[0]


class BngFileInterface(BngBaseInterface):
    def __init__(self, model=None, verbose=False, output_dir=None,
                 output_prefix=None, cleanup=True,
                 model_additional_species=None,
                 model_population_maps=None):
        super(BngFileInterface, self).__init__(
            model, verbose, cleanup, output_prefix, output_dir,
            model_additional_species=model_additional_species,
            model_population_maps=model_population_maps
        )
        self._init_command_queue()

    def _init_command_queue(self):
        """
        Initializes the BNG command queue
        """
        self.command_queue = StringIO()
        self.command_queue.write('begin actions\n')

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        In file interface mode, we close the command queue buffer (whether
        or not it's been executed) and erase the temporary directory if
        applicable.
        """
        self.command_queue.close()
        if self.cleanup:
            self._delete_tmpdir()

    def execute(self, reload_netfile=False, skip_file_actions=True):
        """
        Executes all BNG commands in the command queue.

        Parameters
        ----------
        reload_netfile: bool or str
            If true, attempts to reload an existing .net file from a
            previous execute() iteration. If a string, the filename
            specified in the string is supplied to BNG's readFile (which can be
            any file type BNG supports, such as .net or .bngl).
            This is useful for running multiple actions in a row,
            where results need to be read into PySB before a new series of
            actions is executed.
        skip_file_actions: bool
            Only used if the previous argument is not False. Set this
            argument to True to ignore any actions block in the loaded file.
        """
        self.command_queue.write('end actions\n')
        bng_commands = self.command_queue.getvalue()

        # Generate BNGL file
        with open(self.bng_filename, 'w') as bng_file:
            output = ''
            if self.model and not reload_netfile:
                output += self.generator.get_content()
            if reload_netfile:
                filename = reload_netfile if \
                    isinstance(reload_netfile, basestring) \
                    else self.net_filename
                output += '\n  readFile({file=>"%s",skip_actions=>%d})\n' \
                    % (filename, int(skip_file_actions))
            output += bng_commands
            self._logger.debug('BNG command file contents:\n\n' + output)
            bng_file.write(output)

        # Reset the command queue, in case execute() is called again
        self.command_queue.close()
        self._init_command_queue()

        bng_exec_args = [pf.get_path('bng'), self.bng_filename]
        if not bng_exec_args[0].endswith('.bat'):
            bng_exec_args.insert(0, 'perl')

        p = subprocess.Popen(bng_exec_args,
                             cwd=self.base_directory,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        for line in iter(p.stdout.readline, b''):
            self._logger.debug(line[:-1])
        (p_out, p_err) = p.communicate()
        p_out = p_out.decode('utf-8')
        p_err = p_err.decode('utf-8')
        if p.returncode:
            raise BngInterfaceError(p_out.rstrip("at line") + "\n" +
                                    p_err.rstrip())

    def action(self, action, **kwargs):
        """
        Generates a BNG action command and adds it to the command queue

        Parameters
        ----------
        action : string
            The name of the BNG action function
        kwargs : kwargs, optional
            Arguments and values to supply to BNG
        """
        # Process BNG arguments into a string
        action_args = self._format_action_args(**kwargs)

        # Add the command to the queue
        if action_args == '':
            self.command_queue.write('\t%s()\n' % action)
        else:
            self.command_queue.write('\t%s({%s})\n' % (action, action_args))

        return

    def set_parameter(self, name, value):
        """
        Generates a BNG action command and adds it to the command queue

        Parameters
        ----------
        name: string
            The name of the parameter to set
        value: float-like
            Value of parameter

        """
        self.command_queue.write('\tsetParameter("%s", %g)\n' % (name, value))

    def set_concentration(self, cplx_pat, value):
        """
        Generates a BNG action command and adds it to the command queue

        Parameters
        ----------
        cplx_pat: pysb.ComplexPattern or string
            Species ComplexPattern, or a BNG format string representation
        value: float-like
            Initial concentration

        """
        if isinstance(cplx_pat, basestring):
            formatted_name = cplx_pat
        else:
            formatted_name = format_complexpattern(
                pysb.core.as_complex_pattern(cplx_pat)
            )
        self.command_queue.write('\tsetConcentration("%s", %g)\n' % (
            formatted_name, value))


def generate_hybrid_model(model, population_maps, additional_species=None,
                          safe=False, verbose=False, output_dir=None,
                          output_prefix=None, cleanup=True):
    with BngFileInterface(model,
                          output_dir=output_dir,
                          output_prefix=output_prefix,
                          cleanup=cleanup,
                          model_additional_species=additional_species,
                          model_population_maps=population_maps) as bng:
        bng.action('generate_hybrid_model', verbose=verbose, safe=safe,
                   suffix='hpp')
        bng.execute()

        with open(bng.base_filename + '_hpp.bngl', 'r') as f:
            return f.read()


def run_ssa(model, t_end=10, n_steps=100, param_values=None, output_dir=None,
            output_file_basename=None, cleanup=True, verbose=False,
            **additional_args):
    """
    Simulate a model with BNG's SSA simulator and return the trajectories.

    Parameters
    ----------
    model : Model
        Model to simulate.
    t_end : number, optional
        Final time point of the simulation.
    n_steps : int, optional
        Number of steps in the simulation.
    param_values : vector-like or dictionary, optional
            Values to use for every parameter in the model. Ordering is
            determined by the order of model.parameters. 
            If not specified, parameter values will be taken directly from
            model.parameters.
    output_dir : string, optional
        Location for temporary files generated by BNG. If None (the
        default), uses a temporary directory provided by the system. A
        temporary directory with a random name is created within the
        supplied location.
    output_file_basename : string, optional
        This argument is used as a prefix for the temporary BNG
        output directory, rather than the individual files.
    cleanup : bool, optional
        If True (default), delete the temporary files after the simulation is
        finished. If False, leave them in place. Useful for debugging.
    verbose : bool or int, optional (default: False)
        Sets the verbosity level of the logger. See the logging levels and
        constants from Python's logging module for interpretation of integer
        values. False is equal to the PySB default level (currently WARNING),
        True is equal to DEBUG.
    additional_args: kwargs, optional
        Additional arguments to pass to BioNetGen

    """
    bng_action_debug = verbose if isinstance(verbose, bool) else \
        verbose <= EXTENDED_DEBUG

    additional_args['method'] = 'ssa'
    additional_args['t_end'] = t_end
    additional_args['n_steps'] = n_steps
    additional_args['verbose'] = additional_args.get('verbose',
                                                     bng_action_debug)

    if param_values is not None:
        if len(param_values) != len(model.parameters):
            raise Exception("param_values must be the same length as "
                            "model.parameters")
        for i in range(len(param_values)):
            model.parameters[i].value = param_values[i]

    with BngFileInterface(model, verbose=verbose, output_dir=output_dir,
                          output_prefix=output_file_basename,
                          cleanup=cleanup) as bngfile:
        bngfile.action('generate_network', overwrite=True,
                       verbose=bng_action_debug)
        bngfile.action('simulate', **additional_args)

        bngfile.execute()

        output = bngfile.read_netfile()

        # Parse netfile (in case this hasn't already been done)
        if not model.species:
            _parse_netfile(model, iter(output.split('\n')))

        yfull = bngfile.read_simulation_results()

    return yfull


def generate_network(model, cleanup=True, append_stdout=False,
                     verbose=False, **kwargs):
    """
    Return the output from BNG's generate_network function given a model.

    The output is a BNGL model definition with additional sections 'reactions'
    and 'groups', and the 'species' section expanded to contain all possible
    species. BNG refers to this as a 'net' file.

    Parameters
    ----------
    model : Model
        Model to pass to generate_network.
    cleanup : bool, optional
        If True (default), delete the temporary files after the simulation is
        finished. If False, leave them in place (in `output_dir`). Useful for
        debugging.
    append_stdout : bool, optional
        This option is no longer supported and has been left here for API
        compatibility reasons.
    verbose : bool or int, optional (default: False)
        Sets the verbosity level of the logger. See the logging levels and
        constants from Python's logging module for interpretation of integer
        values. False is equal to the PySB default level (currently WARNING),
        True is equal to DEBUG.
    """
    bng_action_debug = verbose if isinstance(verbose, bool) else \
        verbose <= EXTENDED_DEBUG

    with BngFileInterface(model, verbose=verbose, cleanup=cleanup) as bngfile:
        bngfile._logger.info('Generating reaction network')
        bngfile.action('generate_network', overwrite=True,
                       verbose=bng_action_debug, **kwargs)
        bngfile.execute()

        output = bngfile.read_netfile()

    return output


def load_equations(model, netfile):
    """
    Load model equations from a specified netfile

    Useful for large models where BioNetGen network generation takes a long
    time - the .net file can be saved and reloaded using this function at a
    later date.

    Parameters
    ----------
    model: pysb.Model
        PySB model file
    netfile: str
        BNG netfile
    """
    if model.reactions:
        return
    with open(netfile, 'r') as f:
        _parse_netfile(model, f)


def generate_equations(model, cleanup=True, verbose=False, **kwargs):
    """
    Generate math expressions for reaction rates and species in a model.

    This fills in the following pieces of the model:

    * species
    * reactions
    * reactions_bidirectional
    * observables (just `coefficients` and `species` fields for each element)
    
    Parameters
    ----------
    model : Model
        Model to pass to generate_network.
    cleanup : bool, optional
        If True (default), delete the temporary files after the simulation is
        finished. If False, leave them in place (in `output_dir`). Useful for
        debugging.
    verbose : bool or int, optional (default: False)
        Sets the verbosity level of the logger. See the logging levels and
        constants from Python's logging module for interpretation of integer
        values. False is equal to the PySB default level (currently WARNING),
        True is equal to DEBUG.

    """
    # only need to do this once
    # TODO track "dirty" state, i.e. "has model been modified?"
    #   or, use a separate "math model" object to contain ODEs
    if model.reactions:
        return
    lines = iter(generate_network(model, cleanup=cleanup,
                                  verbose=verbose, **kwargs).split('\n'))
    _parse_netfile(model, lines)


def _parse_netfile(model, lines):
    """ Parse species, rxns and groups from a BNGL net file """
    try:
        while 'begin species' not in next(lines):
            pass
        model.species = []
        while True:
            line = next(lines)
            if 'end species' in line: break
            _parse_species(model, line)

        while 'begin reactions' not in next(lines):
            pass

        reaction_cache = {}
        while True:
            line = next(lines)
            if 'end reactions' in line: break
            _parse_reaction(model, line, reaction_cache=reaction_cache)
        # fix up reactions whose reverse version we saw first
        for r in model.reactions_bidirectional:
            if all(r['reverse']):
                r['reactants'], r['products'] = r['products'], r['reactants']
                r['rate'] *= -1
            # now the 'reverse' value is no longer needed
            del r['reverse']

        while 'begin groups' not in next(lines):
            pass
        while True:
            line = next(lines)
            if 'end groups' in line: break
            _parse_group(model, line)

    except StopIteration as e:
        pass


def _parse_species(model, line):
    """Parse a 'species' line from a BNGL net file."""
    index, species, value = line.strip().split()
    species_compartment_name, complex_string = \
        re.match(r'(?:@(\w+)::)?(.*)', species).groups()
    species_compartment = model.compartments.get(species_compartment_name)
    monomer_strings = complex_string.split('.')
    monomer_patterns = []
    for ms in monomer_strings:
        monomer_name, site_strings, monomer_compartment_name = \
            re.match(r'(\w+)\(([^)]*)\)(?:@(\w+))?', ms).groups()
        site_conditions = {}
        if len(site_strings):
            for ss in site_strings.split(','):
                # FIXME this should probably be done with regular expressions
                if '!' in ss and '~' in ss:
                    site_name, condition = ss.split('~')
                    state, bond = condition.split('!')
                    if bond == '?':
                        bond = pysb.core.WILD
                    elif bond == '!':
                        bond = pysb.core.ANY
                    else:
                        bond = int(bond)
                    condition = (state, bond)
                elif '!' in ss:
                    site_name, condition = ss.split('!', 1)
                    if '!' in condition:
                        condition = [int(c) for c in condition.split('!')]
                    else:
                        condition = int(condition)
                elif '~' in ss:
                    site_name, condition = ss.split('~')
                else:
                    site_name, condition = ss, None
                site_conditions[site_name] = condition
        monomer = model.monomers[monomer_name]
        monomer_compartment = model.compartments.get(monomer_compartment_name)
        # Compartment prefix notation in BNGL means "assign this compartment to
        # all molecules without their own explicit compartment".
        compartment = monomer_compartment or species_compartment
        mp = pysb.core.MonomerPattern(monomer, site_conditions, compartment)
        monomer_patterns.append(mp)

    cp = pysb.core.ComplexPattern(monomer_patterns, None)
    model.species.append(cp)


def _parse_reaction(model, line, reaction_cache):
    """Parse a 'reaction' line from a BNGL net file."""
    (number, reactants, products, rate, rule) = line.strip().split(' ', 4)
    # the -1 is to switch from one-based to zero-based indexing
    reactants = tuple(int(r) - 1 for r in reactants.split(',') if r != '0')
    products = tuple(int(p) - 1 for p in products.split(',') if p != '0')
    rate = rate.rsplit('*')
    (rule_list, unit_conversion) = re.match(
        r'#([\w,\(\)]+)(?: unit_conversion=(.*))?\s*$',
        rule).groups()
    rule_list = rule_list.split(',')  # BNG lists all rules that generate a rxn
    # Support new (BNG 2.2.6-stable or greater) and old BNG naming convention
    # for reverse rules
    rule_name, is_reverse = zip(*[re.subn('^_reverse_|\(reverse\)$', '', r)
                                  for r in rule_list])
    is_reverse = tuple(bool(i) for i in is_reverse)
    r_names = ['__s%d' % r for r in reactants]
    rate_param = [model.parameters.get(r) or model.expressions.get(r) or
                  float(r) for r in rate]
    combined_rate = sympy.Mul(*[sympy.S(t) for t in r_names + rate_param])
    reaction = {
        'reactants': reactants,
        'products': products,
        'rate': combined_rate,
        'rule': rule_name,
        'reverse': is_reverse,
        }
    model.reactions.append(reaction)
    # bidirectional reactions
    key = (reactants, products)
    key_reverse = (products, reactants)
    if key in reaction_cache:
        reaction_bd = reaction_cache.get(key)
        reaction_bd['rate'] += combined_rate
        reaction_bd['rule'] += tuple(r for r in rule_name if r not in
                                     reaction_bd['rule'])
    elif key_reverse in reaction_cache:
        reaction_bd = reaction_cache.get(key_reverse)
        reaction_bd['reversible'] = True
        reaction_bd['rate'] -= combined_rate
        reaction_bd['rule'] += tuple(r for r in rule_name if r not in
                                     reaction_bd['rule'])
    else:
        # make a copy of the reaction dict
        reaction_bd = dict(reaction)
        # default to false until we find a matching reverse reaction
        reaction_bd['reversible'] = False
        reaction_cache[key] = reaction_bd
        model.reactions_bidirectional.append(reaction_bd)


def _parse_group(model, line):
    """Parse a 'group' line from a BNGL net file."""
    # values are number (which we ignore), name, and species list
    values = line.strip().split()
    obs = model.observables[values[1]]
    if len(values) == 3:
        # combination is a comma separated list of [coeff*]speciesnumber
        for product in values[2].split(','):
            terms = product.split('*')
            # if no coeff given (just species), insert a coeff of 1
            if len(terms) == 1:
                terms.insert(0, 1)
            obs.coefficients.append(int(terms[0]))
            # -1 to change to 0-based indexing
            obs.species.append(int(terms[1]) - 1)


class NoInitialConditionsError(RuntimeError):
    """Model initial_conditions is empty."""
    def __init__(self):
        RuntimeError.__init__(self, "Model has no initial conditions or "
                                    "zero-order synthesis rules")


class NoRulesError(RuntimeError):
    """Model rules is empty."""
    def __init__(self):
        RuntimeError.__init__(self, "Model has no rules")
