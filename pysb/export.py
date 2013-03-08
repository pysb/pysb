#!/usr/bin/env python
"""
Module containing a  class for returning the BNGL for a given PySB model. Serves as a
command-line wrapper around ``pysb.generator.bng.BngGenerator``.

Usage
=====

At the command-line, run as follows::

    python -m pysb.export model_name.py bngl > model_name.bngl

or

    export.py model_name.py bngl > model_name.bngl

where ``model_name.py`` contains a PySB model definition (i.e., contains
an instance of ``pysb.core.Model`` instantiated as a global variable). The
generated BNGL will be printed to standard out, allowing it to be inspected
or redirected to another file.

Usage in the Python shell
=========================

To use in a Python shell, import a model::

    from pysb.examples.robertson import model

and import this module::

    from pysb.tools import export_potterswheel

then call the function ``run``, passing the model instance::

    potterswheel_output = export_potterswheel.run(model)

then write the output to a file::

    f = open('robertson.m', 'w')
    f.write(potterswheel_output)
    f.close()

"""

import sys
import os
import re

class Export(object):
    """The base exporter class.
    """

    def __init__(self, model):
        """Constructor. Takes a model."""
        self.model = model

    def export(self):
        """The export method. Must be implemented by any subclass.
        """
        raise NotImplementedError()

# Define a dict listing supported formats and the names of the classes
# implementing their export procedures
formats = {
        'bngl': 'ExportBngl',
        'bng_net': 'ExportBngNet',
        'potterswheel': 'ExportPottersWheel',
        }

def export(model, format):
    """Top-level function for exporting a model to a given format.

    Parameters
    ----------
    model : pysb.core.Model
        The model to export.
    format : string
        A string indicating the desired export format.
    """

    # Import the exporter module. This is done at export runtime to avoid
    # circular imports at module loading
    export_module = __import__('pysb.exporters.' + format, fromlist=[formats[format]])
    export_class = getattr(export_module, formats[format])
    e = export_class(model)
    return e.export()

if __name__ == '__main__':
    # Check the arguments
    if len(sys.argv) <= 2:
        raise Exception("You must specify the filename of a model script " \
                        "and a format for export.")
        # FIXME FIXME Print usage message
    model_filename = sys.argv[1]
    format = sys.argv[2]

    # Make sure that the user has supplied an allowable format
    if format not in formats.keys():
        raise Exception("The format must be one of the following: " +
                ", ".join(formats.keys()) + ".")

    # Sanity checks on filename
    if not os.path.exists(model_filename):
        raise Exception("File '%s' doesn't exist" % model_filename)
    if not re.search(r'\.py$', model_filename):
        raise Exception("File '%s' is not a .py file" % model_filename)
    sys.path.insert(0, os.path.dirname(model_filename))
    model_name = re.sub(r'\.py$', '', os.path.basename(model_filename))
    # import it
    try:
        # FIXME if the model has the same name as some other "real" module
        # which we use, there will be trouble (use the imp package and import
        # as some safe name?)
        model_module = __import__(model_name)
    except StandardError as e:
        print "Error in model script:\n"
        raise
    # grab the 'model' variable from the module
    try:
        model = model_module.__dict__['model']
    except KeyError:
        raise Exception("File '%s' isn't a model file" % model_filename)

    # Export the model
    print export(model, format)
