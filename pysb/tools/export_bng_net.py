#!/usr/bin/env python

from pysb.generator.bng import BngGenerator
from pysb.bng import generate_network_code, pkg_path
import re
import sys
import os
import random
import subprocess
from StringIO import StringIO


def run(model):
    gen = BngGenerator(model)
    output = StringIO()
    bng_filename = '%d_%d_temp.bngl' % (os.getpid(), random.randint(0, 10000))
    net_filename = bng_filename.replace('.bngl', '.net')
    # FIXME this should be factored out in bng.py instead of copy+pasted
    try:
        bng_file = open(bng_filename, 'w')
        bng_file.write(gen.get_content())
        bng_file.write(generate_network_code)
        bng_file.close()
        p = subprocess.Popen(['perl', pkg_path + '/Perl2/BNG2.pl', bng_filename],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p_stdout = p.communicate()[0]
        if p.returncode:
            raise Exception(p_stdout)
        net_file = open(net_filename, 'r')
        output.write("# BioNetGen output:\n#\n#")
        output.write(re.sub(r'\n', r'\n# ', p_stdout))
        output.write('\n')
        output.write(net_file.read())
        net_file.close()
    except Exception as e:
        raise Exception("problem running BNG: " + str(e))
    finally:
        for filename in [bng_filename, net_filename]:
            if os.access(filename, os.F_OK):
                os.unlink(filename)
    return output.getvalue()


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
