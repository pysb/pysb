#!/usr/bin/env python
"""
Docstring for this module
"""
import sys
import os
import re

# Import all of the exporter classes
#from pysb.export.bngl import ExportBngl
#from pysb.export.matlab import ExportMatlab


class Export(object):
    """The base exporter class.
    """

    def __init__(self, model):
        """Constructor. Takes a model."""
        self.model = model

    def export(self):
        """The exporter. Maybe return notimplemented error?"""
        pass

class ExportBngl(Export):
    def export(self):
        return "dummy"

# Define the list of supported formats
formats = {
        'bngl': ExportBngl,
        #'matlab': ExportMatlab,
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
    e = formats[format](model)
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
