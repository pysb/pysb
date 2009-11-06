#from pysb.generator.bng import BngGenerator
import os
import subprocess
import random
import sympy

pkg_path = None

generate_network_code = """
begin actions
generate_network({overwrite=>1});
end actions
"""

def generate_equations(content):
    if pkg_path == None:
        raise Exception('must set pysb.bng.pkg_path to BNG directory')

    #gen = BngGenerator(model)

    bng_filename = '%d_%d_temp.bngl' % (os.getpid(), random.randint(0, 10000))
    net_filename = bng_filename.replace('.bngl', '.net')
    try:
        bng_file = open(bng_filename, 'w')
        bng_file.write(content)
        bng_file.write(generate_network_code)
        bng_file.close()

        subprocess.call(['/usr/bin/perl', pkg_path+'/Perl2/BNG2.pl', bng_filename],
                        stdout=subprocess.PIPE)

        net_file = open(net_filename, 'r')
        while net_file.readline().strip() != 'begin reactions':
            pass
        ydot = {}
        while True:
            line = net_file.readline()
            if line == 'end reactions\n' or line == '': break
            (number, reactants, products, rate, rule) = line.strip().split()
            reactants = reactants.split(',')
            products = products.split(',')
            rate = rate.rsplit('*')
            r_names = ['s' + r for r in reactants]
            combined_rate = sympy.Mul(*[sympy.S(t) for t in r_names + rate]) 
            for p in products:
                p = int(p)
                ydot.setdefault(p, sympy.S(0))
                ydot[p] += combined_rate
            for r in reactants:
                r = int(r)
                ydot.setdefault(r, sympy.S(0))
                ydot[r] -= combined_rate
        net_file.close()
    except Exception as e:
        print "problem running BNG:\n"
        print e
        print "\n"
    finally:
        os.unlink(bng_filename)
        os.unlink(net_filename)

    for i in range(1, max(ydot) + 1):
        ydot.setdefault(i, sympy.S(0))

    return ydot
