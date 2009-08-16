from pysb.generator.bng import BngGenerator
import os
import random

pkg_path = None

generate_network_code = """
begin actions
generate_network({overwrite=>1});
end actions
"""

def generate_equations(model):
    if pkg_path == None:
        raise Exception('must set pysb.bng.pkg_path to BNG directory')

    gen = BngGenerator(model)

    bng_filename = '%d_%d_temp.bngl' % (os.getpid(), random.randint(0, 10000))
    bng_file = open(bng_filename, 'w')
    bng_file.write(gen.content)
    bng_file.write(generate_network_code)
    bng_file.close()

    os.spawnl(os.P_WAIT, pkg_path+'/BNG2.pl', 'BNG2.pl', bng_filename)

    net_filename = bng_filename.replace('.bngl', '.net')
    net_file = open(net_filename, 'r')
    while net_file.readline() != 'begin reactions':
        pass
    while True:
        line = net_file.readline()
        if line == 'end reactions': break
        print "line:", line
    net_file.close()

    #os.unlink(bng_filename)
    #os.unlink(net_filename)
