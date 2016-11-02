from pysb.logging import get_logger
import logging

# Nosetests adds its own logging handler - console_output=False avoids
# duplication
get_logger(level=logging.DEBUG, console_output=False)
