import os
import sysconfig

# Set to False to not utilize the system PATH environment variable
use_path = 'PYSB_PATHFINDER_IGNORE_PATH' not in os.environ

_path_config = {
    'atomizer': {
        'name': 'Atomizer',
        'executable': {
            'posix': 'sbmlTranslator',
            'nt': 'sbmlTranslator.exe'
        },
        'env_var': 'BNGPATH',
        'env_var_subdir': 'bin',
        'search_paths': {
            'posix': ('/usr/local/share/BioNetGen/bin',),
            'nt': ('c:/Program Files/BioNetGen/bin',)
        },
        'conda_install_cmd': 'conda install -c alubbock atomizer'
    },
    'bng': {
        'name': 'BioNetGen',
        'executable': 'BNG2.pl',
        'batch_file': 'BNG2.bat',
        'env_var': 'BNGPATH',
        'search_paths': {
            'posix': ('/usr/local/share/BioNetGen', ),
            'nt': ('c:/Program Files/BioNetGen', )
        },
        'conda_install_cmd': 'conda install -c alubbock bionetgen'
    },
    'kasa': {
        'name': 'KaSa (Kappa)',
        'executable': {
            'posix': 'KaSa',
            'nt': 'KaSa.exe'
        },
        'env_var': 'KAPPAPATH',
        'search_paths': {
            'posix': ('/usr/local/share/KaSa', ),
            'nt': ('c:/Program Files/KaSa', )
        },
        'conda_install_cmd': 'conda install -c alubbock kappa'
    },
    'kasim': {
        'name': 'KaSim (Kappa)',
        'executable': {
            'posix': 'KaSim',
            'nt': 'KaSim.exe'
        },
        'env_var': 'KAPPAPATH',
        'search_paths': {
            'posix': ('/usr/local/share/KaSim',),
            'nt': ('c:/Program Files/KaSim',)
        },
        'conda_install_cmd': 'conda install -c alubbock kappa'
    },
    'cupsoda': {
        'name': 'cupSODA',
        'executable': {
            'posix': 'cupSODA',
            'nt': 'cupsoda.exe'
        },
        'env_var': 'CUPSODAPATH',
        'search_paths': {
            'posix': ('/usr/local/share/cupSODA',),
            'nt': ('c:/Program Files/cupSODA',)
        },
        'conda_install_cmd': 'conda install -c alubbock cupsoda'
    },
    'stochkit_ssa': {
        'name': 'StochKit [SSA]',
        'executable': {
            'posix': 'ssa',
            'nt': 'ssa.exe'
        },
        'batch_file': 'ssa.bat',
        'env_var': 'STOCHKITPATH',
        'search_paths': {
            'posix': ('/usr/local/share/StochKit', ),
            'nt': ('c:/Program Files/StochKit',)
        },
        'conda_install_cmd': 'conda install -c alubbock stochkit'
    },
    'stochkit_tau_leaping': {
        'name': 'StochKit [Tau Leaping]',
        'executable': {
            'posix': 'tau_leaping',
            'nt': 'tau_leaping.exe'
        },
        'batch_file': 'tau_leaping.bat',
        'env_var': 'STOCHKITPATH',
        'search_paths': {
            'posix': ('/usr/local/share/StochKit',),
            'nt': ('c:/Program Files/StochKit',)
        },
        'conda_install_cmd': 'conda install -c alubbock stochkit'
    }
}
_path_cache = {}


def list_programs():
    """
    Return the list of available external programs as a dictionary

    Returns
    -------
    A dictionary containing the internal program name (key) and the
    human-readable name and environment variable (value) to adjust the path
    for that program.

    """
    keep_keys = ('name', 'env_var')
    return {prog_name: {
        k: v for k, v in prog_data.items() if k in keep_keys
    } for prog_name, prog_data in _path_config.items()}


def get_path(prog_name):
    """
    Gets the currently active path to an external executable

    The path will be determined automatically if not set (see return value).
    To override, call :func:`set_path`.

    Parameters
    ----------
    prog_name: str
        The PySB internal program name for an executable (run
        :func:`list_programs` for a list).

    Returns
    -------
    The currently active path to an external executable. If the path hasn't
    previously been set, the relevant environment variable for that
    program's path will be searched. Failing that, a list of default paths
    for the operating system will be checked. An Exception is raised if none
    of these approaches works.
    """
    try:
        return _path_cache[prog_name]
    except KeyError:
        pass

    if prog_name not in _path_config.keys():
        raise ValueError('%s is not a known external executable' % prog_name)

    path_conf = _path_config[prog_name]

    # Try environment variable, if set
    if path_conf['env_var'] in os.environ:
        env_var_val = os.environ[path_conf['env_var']]
        subdir_msg = ''
        try:
            _path_cache[prog_name] = _validate_path(prog_name, env_var_val)
            return _path_cache[prog_name]
        except ValueError:
            try:
                _path_cache[prog_name] = _validate_path(
                    prog_name, os.path.join(env_var_val,
                                            path_conf['env_var_subdir']))
                return _path_cache[prog_name]
            except KeyError:
                # No subdirectory set
                pass
            except ValueError:
                # Subdirectory set, but no binary found
                subdir_msg = ', or in that path\'s "%s" subdirectory' %\
                             path_conf['env_var_subdir']
            raise ValueError('Environment variable %s is set to %s, but the '
                             'program %s or its executable %s could not be '
                             'found there%s. Check file existence and '
                             'permissions.' % (
                                path_conf['env_var'],
                                env_var_val,
                                path_conf['name'],
                                _get_executable(prog_name),
                                subdir_msg)
                             )

    # Check the Anaconda environment, if applicable, or BINDIR
    try:
        _path_cache[prog_name] = _validate_path(prog_name,
                                                _get_anaconda_bindir())
        return _path_cache[prog_name]
    except ValueError:
        pass

    # Check default paths for this operating system
    if os.name not in path_conf['search_paths'].keys():
        raise Exception('No default path is known for %s on your '
                        'operating system "%s". Set the path using the '
                        'environment variable %s or by calling the function '
                        '%s.%s()' % (path_conf['name'],
                                     os.name,
                                     path_conf['env_var'],
                                     set_path.__module__,
                                     set_path.__name__))

    search_paths = path_conf['search_paths'][os.name]
    if use_path:
        search_paths = list(search_paths) + os.environ.get('PATH', '').split(
            os.pathsep)

    for search_path in search_paths:
        try:
            _path_cache[prog_name] = _validate_path(prog_name, search_path)
            return _path_cache[prog_name]
        except ValueError:
            pass

    try:
        conda_install_help = '\n\nConda users can install %s using the ' \
                             'following command:\n\n%s' % \
                             (path_conf['name'], path_conf['conda_install_cmd'])
    except KeyError:
        conda_install_help = ''

    raise Exception('The program %s was not found in the default search '
                    'path(s) for your operating system:\n\n%s\n\nEither '
                    'install it to one of those paths, or set a custom path '
                    'using the environment variable %s or by calling the '
                    'function %s.%s()%s' % (path_conf['name'],
                                            "\n".join(search_paths),
                                            path_conf['env_var'],
                                            set_path.__module__,
                                            set_path.__name__,
                                            conda_install_help)
                    )


def set_path(prog_name, full_path):
    """
    Sets the full path to an external executable at runtime

    External program paths can also be adjusted by environment variable prior
    to first use; run :func:`list_programs` for a list of programs.

    Parameters
    ----------
    prog_name: str
        The internal program name for an executable. (see
        :func:`list_programs`)
    full_path: str
        The full path to the external executable or its enclosing directory.
        If the path is a directory, it will be searched for the executable.
        A ValueError will be raised if there's an issue with the path (not
        found, permissions etc.).
    """
    if prog_name not in _path_config.keys():
        raise ValueError('%s is not a known external executable' % prog_name)

    _path_cache[prog_name] = _validate_path(prog_name, full_path)



def _get_anaconda_bindir():
    """ Get the binary path from python build time (for anaconda) """
    # Is this an anaconda virtual environment?
    conda_env = os.environ.get('CONDA_PREFIX', None)
    if conda_env:
        return os.path.join(conda_env, 'Scripts' if os.name == 'nt' else 'bin')

    # Otherwise, try the default anaconda/python bin directory
    bindir = sysconfig.get_config_var('BINDIR')
    if os.name == 'nt':
        # bindir doesn't point to scripts directory on Windows
        return os.path.join(bindir, 'Scripts')
    else:
        return bindir

def _get_batch_file(prog_name):
    try:
        return _path_config[prog_name]['batch_file']
    except KeyError:
        return None


def _get_executable(prog_name):
    executable = _path_config[prog_name]['executable']
    if isinstance(executable, str):
        return executable
    else:
        try:
            return executable[os.name]
        except KeyError:
            raise Exception('No executable for "%s" is available for your '
                            'operating system: "%s"' %
                            (_path_config[prog_name]['name'],
                             os.name))


def _validate_path(prog_name, full_path):
    if not os.access(full_path, os.F_OK):
        raise ValueError('Unable to access path %s. Check the file exists '
                         'and the current user has permission to access it.' %
                         full_path)

    if not os.path.isfile(full_path):
        # On anaconda, check batch file on Windows, if applicable
        batch_file = _get_batch_file(prog_name)
        if os.name == 'nt' and batch_file:
            try:
                return _validate_path(prog_name,
                                      os.path.join(full_path, batch_file))
            except ValueError:
                pass

        # It's a directory, try appending the executable name
        return _validate_path(prog_name, os.path.join(full_path,
                                                      _get_executable(
                                                          prog_name)))

    if not os.access(full_path, os.X_OK):
        raise ValueError('The file %s does not have executable permissions.' %
                         full_path)

    return full_path
