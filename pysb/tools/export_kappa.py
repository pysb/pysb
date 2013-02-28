#!/usr/bin/env python
"""
A module for returning the Kappa equivalent for a given PySB model.  Serves as
a command-line wrapper around ``pysb.generator.kappa.KappaGenerator``.

Note that the Kappa syntax for the KaSim simulator is slightly different
from that for the complx analyzer; this script generates syntax for the
KaSim simulator.

Usage
=====

At the command-line, run as follows::

    export_kappa.py model_name.py > model_name.ka

where ``model_name.py`` contains a PySB model definition (i.e., contains
an instance of ``pysb.core.Model`` instantiated as a global variable). The
generated Kappa will be printed to standard out, allowing it to be inspected
or redirected to another file.
"""

from pysb.generator.kappa import KappaGenerator
import re
import sys
import os

def run(model, dialect='kasim'):
    """Generate the corresponding Kappa for the given PySB model.
    A wrapper around ``pysb.generator.kappa.KappaGenerator``.

    Parameters
    ----------
    model : pysb.core.Model
        The model to generate Kappa for.
    dialect : string, either 'kasim' or 'complx'
        The Kappa file syntax for the Kasim simulator is slightly
        different from that of the complx analyzer. This argument
        specifies which type of Kappa to produce ('kasim' is the default).

    Returns
    -------
    string
        The Kappa output.
    """

    gen = KappaGenerator(model, dialect=dialect)
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
    print run(model)
