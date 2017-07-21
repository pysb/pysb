import os

_path_config = {
    'bng': {
        'name': 'BioNetGen',
        'executable': 'BNG2.pl',
        'env_var': 'BNGPATH',
        'search_paths': {
            'posix': ('/usr/local/share/BioNetGen', ),
            'nt': ('c:/Program Files/BioNetGen', )
        }
    },
    'kasa': {
        'name': 'KaSa (Kappa)',
        'executable': 'KaSa',
        'env_var': 'KAPPAPATH',
        'search_paths': {
            'posix': ('/usr/local/share/KaSa', ),
            'nt': ('c:/Program Files/KaSa', )
        }
    },
    'kasim': {
        'name': 'KaSim (Kappa)',
        'executable': 'KaSim',
        'env_var': 'KAPPAPATH',
        'search_paths': {
            'posix': ('/usr/local/share/KaSim',),
            'nt': ('c:/Program Files/KaSim',)
        }
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
        }
    },
    'stochkit_ssa': {
        'name': 'StochKit [SSA]',
        'executable': 'ssa',
        'env_var': 'STOCHKITPATH',
        'search_paths': {
            'posix': ('/usr/local/share/StochKit', ),
            'nt': ('c:/Program Files/StochKit',)
        }
    },
    'stochkit_tau_leaping': {
        'name': 'StochKit [Tau Leaping]',
        'executable': 'tau_leaping',
        'env_var': 'STOCHKITPATH',
        'search_paths': {
            'posix': ('/usr/local/share/StochKit',),
            'nt': ('c:/Program Files/StochKit',)
        }
    },
}
_path_cache = {}


def get_path(prog_name):
    """
    Gets the currently active path to an external executable

    The path will be determined automatically if not set (see return value).
    To override, call :func:`set_path`.

    Parameters
    ----------
    prog_name: str
        The PySB internal program name for an executable. One of 'bng'
        (BioNetGen), 'kasa' (Kappa's KaSa) or 'kasim' (Kappa's KaSim).

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
        try:
            _path_cache[prog_name] = _validate_path(prog_name, env_var_val)
            return _path_cache[prog_name]
        except ValueError:
            raise ValueError('Environment variable %s is set to %s, but the '
                             'program %s or its executable %s could not be '
                             'found there. Check file existence and '
                             'permissions.' % (
                                path_conf['env_var'],
                                env_var_val,
                                path_conf['name'],
                                _get_executable(prog_name)))

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
    for search_path in search_paths:
        try:
            _path_cache[prog_name] = _validate_path(prog_name, search_path)
            return _path_cache[prog_name]
        except ValueError:
            pass

    raise Exception('The program %s was not found in the default search '
                    'path(s) for your operating system:\n\n%s\n\nEither '
                    'install it to one of those paths, or set a custom path '
                    'using the environment variable %s or by calling the '
                    'function %s.%s()' % (path_conf['name'],
                                          "\n".join(search_paths),
                                          path_conf['env_var'],
                                          set_path.__module__,
                                          set_path.__name__))


def set_path(prog_name, full_path):
    """
    Sets the full path to an external executable

    Parameters
    ----------
    prog_name: str
        The internal program name for an executable. See :func:`get_path` for
        valid values.
    full_path: str
        The full path to the external executable or its enclosing directory.
        If the path is a directory, it will be searched for the executable.
        A ValueError will be raised if there's an issue with the path (not
        found, permissions etc.).
    """
    if prog_name not in _path_config.keys():
        raise ValueError('%s is not a known external executable' % prog_name)

    _path_cache[prog_name] = _validate_path(prog_name, full_path)


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
        # It's a directory, try appending the executable name
        return _validate_path(prog_name, os.path.join(full_path,
                                                      _get_executable(
                                                          prog_name)))

    if not os.access(full_path, os.X_OK):
        raise ValueError('The file %s does not have executable permissions.' %
                         full_path)

    return full_path
