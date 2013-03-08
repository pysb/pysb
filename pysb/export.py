#!/usr/bin/env python
"""
A script for exporting PySB models to a variety of other formats.

Exporting can be performed at the command-line or programmatically/interactively
from within Python.

Command-line usage
==================

At the command-line, run as follows::

    python -m pysb.export model_name.py [format]

where ``model_name.py`` is a file containing a PySB model definition (i.e.,
contains an instance of ``pysb.core.Model`` instantiated as a global variable).
``[format]`` should be the name of one of the supported formats:

- ``bngl``
- ``bng_net``
- ``kappa``
- ``potterswheel``
- ``sbml``
- ``python``
- ``mathematica``
- ``matlab``

In all cases, the exported model code will be printed to standard
out, allowing it to be inspected or redirected to another file.

Interactive usage
=================

Export functionality is implemented by this module's top-level function
``export``. For example, to export the "Robertson" example model as SBML, first
import the model::

    from pysb.examples.robertson import model

Then import the ``export`` function from this module::

    from pysb.export import export

Call the ``export`` function, passing the model instance and a string
indicating the desired format, which should be one of the ones indicated
in the list in the "Command-line usage" section above::

    sbml_output = export(model, 'sbml')

The output (a string) can be inspected or written to a file, e.g. as follows::

    with open('robertson.sbml', 'w') as f:
        f.write(sbml_output)


Implementation of specific exporters
====================================

Information on the implementation of specific exporters can be found in the
documentation for the exporter classes in the package :py:mod:`pysb.exporters`:

.. toctree::
   :maxdepth: 2

   exporters/sbml.rst
   exporters/matlab.rst
   exporters/mathematica.rst
   exporters/potterswheel.rst
   exporters/bngl.rst
   exporters/bng_net.rst
   exporters/kappa.rst
   exporters/python.rst
"""

import sys
import os
import re

class Export(object):
    """Base class for all PySB model exporters.

    Export functionality is implemented by subclasses of this class. The
    pattern for model export is the same for all exporter subclasses: a
    model is passed to the exporter constructor and the ``export`` method
    on the instance is called.

    Parameters
    ----------
    model : pysb.core.Model
        The model to export.

    Examples
    --------

    Exporting the "Robertson" example model to SBML using the ``ExportSbml``
    subclass::

    >>> from pysb.examples.robertson import model
    >>> from pysb.exporters.sbml import ExportSbml
    >>> e = ExportSbml(model)
    >>> sbml_output = e.export()
    """

    def __init__(self, model):
        self.model = model
        """The model to export."""

    def export(self):
        """The export method, which must be implemented by any subclass.

        All implementations of this method are expected to return a single
        string containing the representation of the model in the desired
        format.
        """
        raise NotImplementedError()

# Define a dict listing supported formats and the names of the classes
# implementing their export procedures
formats = {
        'bngl': 'ExportBngl',
        'bng_net': 'ExportBngNet',
        'kappa': 'ExportKappa',
        'potterswheel': 'ExportPottersWheel',
        'sbml': 'ExportSbml',
        'python': 'ExportPython',
        'mathematica': 'ExportMathematica',
        'matlab': 'ExportMatlab',
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
    export_module = __import__('pysb.exporters.' + format,
                               fromlist=[formats[format]])
    export_class = getattr(export_module, formats[format])
    e = export_class(model)
    return e.export()

if __name__ == '__main__':
    # Check the arguments
    if len(sys.argv) <= 2:
        print __doc__,
        exit()

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
