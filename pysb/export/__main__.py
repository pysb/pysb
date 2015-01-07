from __future__ import print_function
import os
import sys
import re
import pysb.export


def validate_argv(argv):
    return len(argv) == 3


def main(argv):
    if not validate_argv(argv):
        print(pysb.export.__doc__, end=' ')
        return 1

    model_filename = argv[1]
    format = argv[2]

    # Make sure that the user has supplied an allowable format
    if format not in pysb.export.formats.keys():
        raise Exception("The format must be one of the following: " +
                ", ".join(pysb.export.formats.keys()) + ".")

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
    except Exception as e:
        print("Error in model script:\n")
        raise
    # grab the 'model' variable from the module
    try:
        model = model_module.__dict__['model']
    except KeyError:
        raise Exception("File '%s' isn't a model file" % model_filename)

    # Export the model
    print(pysb.export.export(model, format, model_module.__doc__))

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
