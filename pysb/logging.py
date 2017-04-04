from __future__ import absolute_import
import logging
import platform
import socket
import time
import os
import warnings
import pysb

SECONDS_IN_HOUR = 3600
LOG_LEVEL_ENV_VAR = 'PYSB_LOG'
BASE_LOGGER_NAME = 'pysb'
EXTENDED_DEBUG = 5
NAMED_LOG_LEVELS = {'NOTSET': logging.NOTSET,
                    'EXTENDED_DEBUG': EXTENDED_DEBUG,
                    'DEBUG': logging.DEBUG,
                    'INFO': logging.INFO,
                    'WARNING': logging.WARNING,
                    'ERROR': logging.ERROR,
                    'CRITICAL': logging.CRITICAL}


def formatter(time_utc=False):
    """
    Build a logging formatter using local or UTC time

    Parameters
    ----------
    time_utc : bool, optional (default: False)
        Use local time stamps in log messages if False, or Universal
        Coordinated Time (UTC) if True

    Returns
    -------
    A logging.Formatter object for PySB logging
    """
    log_fmt = logging.Formatter('%(asctime)s.%(msecs).3d - %(name)s - '
                                '%(levelname)s - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
    if time_utc:
        log_fmt.converter = time.gmtime

    return log_fmt


def setup_logger(level=logging.WARNING, console_output=True, file_output=False,
                 time_utc=False, capture_warnings=True):
    """
    Set up a new logging.Logger for PySB logging

    Calling this method will override any existing handlers, formatters etc.
    attached to the PySB logger. Typically, :func:`get_logger` should be
    used instead, which returns the existing PySB logger if it has already
    been set up, and can handle PySB submodule namespaces.

    Parameters
    ----------
    level : int
        Logging level, typically using a constant like logging.INFO or
        logging.DEBUG
    console_output : bool
        Set up a default console log handler if True (default)
    file_output : string
        Supply a filename to copy all log output to that file, or set to
        False to disable (default)
    time_utc : bool
        Specifies whether log entry time stamps should be in local time
        (when False) or UTC (True). See :func:`formatter` for more
        formatting options.
    capture_warnings : bool
        Capture warnings from Python's warnings module if True (default)

    Returns
    -------
    A logging.Logger object for PySB logging. Note that other PySB modules
    should use a logger specific to their namespace instead by calling
    :func:`get_logger`.
    """
    log = logging.getLogger(BASE_LOGGER_NAME)

    # Logging level can be overridden with environment variable
    if LOG_LEVEL_ENV_VAR in os.environ:
        try:
            level = int(os.environ[LOG_LEVEL_ENV_VAR])
        except ValueError:
            # Try parsing as a name
            level_name = os.environ[LOG_LEVEL_ENV_VAR]
            if level_name in NAMED_LOG_LEVELS.keys():
                level = NAMED_LOG_LEVELS[level_name]
            else:
                raise ValueError('Environment variable {} contains an '
                                 'invalid value "{}". If set, its value must '
                                 'be one of {} (case-sensitive) or an '
                                 'integer log level.'.format(
                    LOG_LEVEL_ENV_VAR, level_name,
                    ", ".join(NAMED_LOG_LEVELS.keys())
                                 ))

    log.setLevel(level)

    # Remove default logging handler
    log.handlers = []

    log_fmt = formatter(time_utc=time_utc)

    if console_output:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_fmt)
        log.addHandler(stream_handler)

    if file_output:
        file_handler = logging.FileHandler(file_output)
        file_handler.setFormatter(log_fmt)
        log.addHandler(file_handler)

    log.info('Logging started on PySB version %s', pysb.__version__)
    if time_utc:
        log.info('Log entry times are in UTC')
    else:
        utc_offset = time.timezone if (time.localtime().tm_isdst == 0) else \
            time.altzone
        utc_offset = -(utc_offset / SECONDS_IN_HOUR)
        log.info('Log entry time offset from UTC: %.2f hours', utc_offset)

    log.debug('OS Platform: %s', platform.platform())
    log.debug('Python version: %s', platform.python_version())
    log.debug('Hostname: %s', socket.getfqdn())

    logging.captureWarnings(capture_warnings)

    return log


def get_logger(logger_name=BASE_LOGGER_NAME, model=None, log_level=None,
               **kwargs):
    """
    Returns (if extant) or creates a PySB logger

    If the PySB base logger has already been set up, this method will return it
    or any of its descendant loggers without overriding the settings - i.e.
    any values supplied as kwargs will be ignored.

    Parameters
    ----------
    logger_name : string
        Get a logger for a specific namespace, typically __name__ for code
        outside of classes or self.__module__ inside a class
    model : pysb.Model
        If this logger is related to a specific model instance, pass the
        model object as an argument to have the model's name prepended to
        log entries
    log_level : bool or int
        Override the default or preset log level for the requested logger.
        None or False uses the default or preset value. True evaluates to
        logging.DEBUG. Any integer is used directly.
    **kwargs : kwargs
        Keyword arguments to supply to :func:`setup_logger`. Only used when
        the PySB logger hasn't been set up yet (i.e. there have been no
        calls to this function or :func:`get_logger` directly).

    Returns
    -------
    A logging.Logger object with the requested name

    Examples
    --------

    >>> from pysb.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.debug('Test message')
    """
    if BASE_LOGGER_NAME not in logging.Logger.manager.loggerDict.keys():
        setup_logger(**kwargs)
    elif kwargs:
        warnings.warn('PySB logger already exists, ignoring keyword '
                      'arguments to setup_logger')

    logger = logging.getLogger(logger_name)

    if log_level is not None and log_level is not False:
        if isinstance(log_level, bool):
            log_level = logging.DEBUG
        elif not isinstance(log_level, int):
            raise ValueError('log_level must be a boolean, integer or None')

        if logger.getEffectiveLevel() != log_level:
            logger.debug('Changing log_level from %d to %d' % (
                logger.getEffectiveLevel(), log_level))
            logger.setLevel(log_level)

    if model is None:
        return logger
    else:
        return PySBModelLoggerAdapter(logger, {'model': model})


class PySBModelLoggerAdapter(logging.LoggerAdapter):
    """ A logging adapter to prepend a model's name to log entries """
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['model'].name, msg), kwargs
