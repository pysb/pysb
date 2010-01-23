import pysb
from pysb.generator.bng import BngGenerator
import os
import subprocess
import random
import re
import sympy

# not ideal, but it will work for now during development
pkg_path = '/usr/local/share/bionetgen'

generate_network_code = """
begin actions
generate_network({overwrite=>1});
end actions
"""

def generate_equations(model):
    gen = BngGenerator(model)

    bng_filename = '%d_%d_temp.bngl' % (os.getpid(), random.randint(0, 10000))
    net_filename = bng_filename.replace('.bngl', '.net')
    try:
        bng_file = open(bng_filename, 'w')
        bng_file.write(gen.get_content())
        bng_file.write(generate_network_code)
        bng_file.close()
        subprocess.call(['/usr/bin/perl', pkg_path+'/Perl2/BNG2.pl', bng_filename],
                        stdout=subprocess.PIPE)
        net_file = open(net_filename, 'r')
        lines = iter(net_file.readlines())
        net_file.close()
    except Exception as e:
        print "problem running BNG:\n"
        print e
        print "\n"
    finally:
        os.unlink(bng_filename)
        os.unlink(net_filename)

    try:
        while 'begin species' not in lines.next():
            pass
        model.species = []
        while True:
            line = lines.next()
            if 'end species' in line: break
            _parse_species(model, line)

        while 'begin reactions' not in lines.next():
            pass
        model.odes = [sympy.S(0)] * len(model.species)
        while True:
            line = lines.next()
            if 'end reactions' in line: break
            (number, reactants, products, rate, rule) = line.strip().split()
            reactants = reactants.split(',')
            products = products.split(',')
            rate = rate.rsplit('*')
            # note that the -1 here and below is to switch to zero-based indexing
            r_names = ['s%d' % (int(r) - 1) for r in reactants]
            combined_rate = sympy.Mul(*[sympy.S(t) for t in r_names + rate]) 
            for p in products:
                model.odes[int(p) - 1] += combined_rate
            for r in reactants:
                model.odes[int(r) - 1] -= combined_rate

        while 'begin groups' not in lines.next():
            pass
        model.observable_groups = {}
        while True:
            line = lines.next()
            if 'end groups' in line: break
            _parse_group(model, line)

    except StopIteration as e:
        pass


def _parse_species(model, line):
    (index, complex_string, value) = line.strip().split()
    monomer_strings = complex_string.split('.')
    monomer_patterns = []
    for ms in monomer_strings:
        monomer_name, site_strings = re.match(r'(\w+)\(([^)]*)\)', ms).groups()
        site_conditions = {}
        for ss in site_strings.split(','):
            if '!' in ss:
                site_name, condition = ss.split('!')
                condition = int(condition)
            elif '~' in ss:
                site_name, condition = ss.split('~')
            else:
                site_name, condition = ss, None
            site_conditions[site_name] = condition
        monomer = model.get_monomer(monomer_name)
        monomer_patterns.append(monomer(site_conditions))

    cp = pysb.ComplexPattern(monomer_patterns)
    model.species.append(cp)


def _parse_group(model, line):
    (number, name, combination) = line.strip().split()
    group = []
    # combination is a comma separated list of [factor*]speciesnumber
    for product in combination.split(','):
        terms = product.split('*')
        # if no factor given (just species), insert a factor of 1
        if len(terms) == 1:
            terms.insert(0, 1)
        factor = int(terms[0])
        species = int(terms[1]) - 1  # -1 to change to 0-based indexing
        group.append((factor, species))  
    model.observable_groups[name] = group
