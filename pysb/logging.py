from __future__ import absolute_import
import logging
import platform
import socket
import time
import os
import warnings
import pysb

SECONDS_IN_HOUR = 3600
DEBUG_ENV_VAR = 'PYSB_DEBUG'
BASE_LOGGER_NAME = 'pysb'


def formatter(time_utc=False):
    """
    Build a logging formatter using local or UTC time

    Parameters
    ----------
    time_utc : bool, optional (default: False)
        Use local time stamps in log messages if False, or Universal
        Coordinated Time (UTC), also known as Greenwich Mean Time (GMT) if True

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


def setup_logger(level=logging.INFO, console_output=True, file_output=False,
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
    if DEBUG_ENV_VAR in os.environ:
        level = logging.DEBUG
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


def get_logger(logger_name=BASE_LOGGER_NAME, **kwargs):
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

    return logging.getLogger(logger_name)
