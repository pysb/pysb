"""
Tools for exporting PySB models to a variety of other formats.

Exporting can be performed at the command-line or programmatically/interactively
from within Python.

Command-line usage
==================

At the command-line, run as follows::

    python -m pysb.export model.py <format>

where ``model.py`` is a file containing a PySB model definition (i.e.,
contains an instance of ``pysb.core.Model`` instantiated as a global variable).
``[format]`` should be the name of one of the supported formats:

- ``bngl``
- ``bng_net``
- ``kappa``
- ``potterswheel``
- ``sbml``
- ``python``
- ``pysb_flat``
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
documentation for the exporter classes in the package :py:mod:`pysb.export`:

.. toctree::
   :maxdepth: 2

   sbml
   matlab
   mathematica
   potterswheel
   bngl
   bng_net
   kappa
   python
   pysb_flat
"""

import re
import textwrap

class Exporter(object):
    """Base class for all PySB model exporters.

    Export functionality is implemented by subclasses of this class. The
    pattern for model export is the same for all exporter subclasses: a
    model is passed to the exporter constructor and the ``export`` method
    on the instance is called.

    Parameters
    ----------
    model : pysb.core.Model
        The model to export.
    docstring : string (optional)
        The header comment to include at the top of the exported file.

    Examples
    --------

    Exporting the "Robertson" example model to SBML using the ``SbmlExporter``
    subclass::

    >>> from pysb.examples.robertson import model
    >>> from pysb.export.sbml import SbmlExporter
    >>> e = SbmlExporter(model)
    >>> sbml_output = e.export()
    """

    def __init__(self, model, docstring=None):
        self.model = model
        """The model to export."""
        self.docstring = docstring
        """Header comment to include at the top of the exported file."""

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
        'bngl': 'BnglExporter',
        'bng_net': 'BngNetExporter',
        'kappa': 'KappaExporter',
        'potterswheel': 'PottersWheelExporter',
        'sbml': 'SbmlExporter',
        'python': 'PythonExporter',
        'pysb_flat': 'PysbFlatExporter',
        'mathematica': 'MathematicaExporter',
        'matlab': 'MatlabExporter',
        }

def export(model, format, docstring=None):
    """Top-level function for exporting a model to a given format.

    Parameters
    ----------
    model : pysb.core.Model
        The model to export.
    format : string
        A string indicating the desired export format.
    docstring : string (optional)
        The header comment to include at the top of the exported file.
    """

    # Import the exporter module. This is done at export runtime to avoid
    # circular imports at module loading
    export_module = __import__('pysb.export.' + format,
                               fromlist=[formats[format]])
    export_class = getattr(export_module, formats[format])
    e = export_class(model, docstring)
    return e.export()

def pad(text, depth=0):
    "Dedent multi-line string and pad with spaces."
    text = textwrap.dedent(text)
    text = re.sub(r'^(?m)', ' ' * depth, text)
    text += '\n'
    return text
