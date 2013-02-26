#!/usr/bin/env python
"""
A module for returning the BNGL for a given PySB model. Can be used as a
command-line script or from within the Python shell. Essentially serves
as a wrapper around ``pysb.generator.bng.BngGenerator``.

Usage as a command-line script
==============================

As a command-line script, run as follows::

    export_bng.py model_name.py > model_name.bngl

where ``model_name.py`` contains a PySB model definition (i.e., contains
an instance of ``pysb.core.Model`` instantiated as a global variable). The
generated BNGL will be printed to standard out, allowing it to be inspected
or redirected to another file.

Usage in the Python shell
=========================

To use in the Python shell, follow this pattern::

    from pysb.examples.robertson import model
    from pysb.tools import export_bng
    bngl_output = export_bng.run(model)
"""

from pysb.generator.bng import BngGenerator
import re
import sys
import os


def run(model):
    """Generate the corresponding BNGL for the given PySB model.
    A wrapper around ``pysb.generator.bng.BngGenerator``.

    Parameters
    ----------
    model : pysb.core.Model
        The model to generate BNGL for.

    Returns
    -------
    string
        The BNGL output.
    """

    gen = BngGenerator(model)
    return gen.get_content()


if __name__ == '__main__':
    # sanity checks on filename
    if len(sys.argv) <= 1:
        raise Exception("You must specify the filename of a model script")
    model_filename = sys.argv[1]
    if not os.path.exists(model_filename):
        raise Exception("File '%s' doesn't exist" % model_filename)
    if not re.search(r'\.py$', model_filename):
        raise Exception("File '%s' is not a .py file" % model_filename)
    sys.path.insert(0, os.path.dirname(model_filename))
    model_name = re.sub(r'\.py$', '', os.path.basename(model_filename))
    # import it
    try:
        # FIXME if the model has the same name as some other "real" module which we use,
        # there will be trouble (use the imp package and import as some safe name?)
        model_module = __import__(model_name)
    except StandardError as e:
        print "Error in model script:\n"
        raise
    # grab the 'model' variable from the module
    try:
        model = model_module.__dict__['model']
    except KeyError:
        raise Exception("File '%s' isn't a model file" % model_filename)
    print run(model)
