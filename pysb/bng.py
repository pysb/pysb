import pysb
from pysb.generator.bng import BngGenerator
import os
import subprocess
import random
import re
import sympy

# not ideal, but it will work for now during development
pkg_path = None
pkg_paths_to_check = [
    '/usr/local/share/BioNetGen',
    'c:/Program Files/BioNetGen'
    ]
for test_path in pkg_paths_to_check:
    if os.access(test_path + '/Perl2/BNG2.pl', os.F_OK):
        pkg_path = test_path
        break
if pkg_path is None:
    msg = "Could not find BioNetGen installed in one of the following locations:\n    " + \
        '\n    '.join(pkg_paths_to_check)
    raise Exception(msg)

generate_network_code = """
begin actions
generate_network({overwrite=>1});
end actions
"""

def generate_equations(model):
    # only need to do this once
    # TODO track "dirty" state, i.e. "has model been modified?"
    #   or, use a separate "math model" object to contain ODEs
    if model.odes:
        return

    gen = BngGenerator(model)

    bng_filename = '%d_%d_temp.bngl' % (os.getpid(), random.randint(0, 10000))
    net_filename = bng_filename.replace('.bngl', '.net')
    try:
        bng_file = open(bng_filename, 'w')
        bng_file.write(gen.get_content())
        bng_file.write(generate_network_code)
        bng_file.close()
        p = subprocess.Popen(['perl', pkg_path + '/Perl2/BNG2.pl', bng_filename],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.communicate()
        if p.returncode:
            raise Exception(p.stdout.read())
        net_file = open(net_filename, 'r')
        lines = iter(net_file.readlines())
        net_file.close()
    except Exception as e:
        raise Exception("problem running BNG: " + str(e))
    finally:
        for filename in [bng_filename, net_filename]:
            if os.access(filename, os.F_OK):
                os.unlink(filename)

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
        compartment_name, monomer_name, site_strings = re.match(r'(?:@(\w+)::)?(\w+)\(([^)]*)\)', ms).groups()
        site_conditions = {}
        if len(site_strings):
            for ss in site_strings.split(','):
                # FIXME this should probably be done with regular expressions
                if '!' in ss and '~' in ss:
                    site_name, condition = ss.split('~')
                    state, bond = condition.split('!')
                    if bond == '?':
                        bond = pysb.WILD
                    elif bond == '!':
                        bond = pysb.ANY
                    else:
                        bond = int(bond)
                    condition = (state, bond)
                elif '!' in ss:
                    site_name, condition = ss.split('!', 1)
                    if '!' in condition:
                        condition = [int(c) for c in condition.split('!')]
                    else:
                        condition = int(condition)
                elif '~' in ss:
                    site_name, condition = ss.split('~')
                else:
                    site_name, condition = ss, None
                site_conditions[site_name] = condition
        monomer = model.monomers[monomer_name]
        monomer_patterns.append(monomer(site_conditions))

    compartment = model.compartments.get(compartment_name)
    cp = pysb.ComplexPattern(monomer_patterns, compartment)
    model.species.append(cp)


def _parse_group(model, line):
    # values are number (which we ignore), name, and species list
    values = line.strip().split()
    group = []
    if len(values) == 3:
        # combination is a comma separated list of [factor*]speciesnumber
        for product in values[2].split(','):
            terms = product.split('*')
            # if no factor given (just species), insert a factor of 1
            if len(terms) == 1:
                terms.insert(0, 1)
            factor = int(terms[0])
            species = int(terms[1]) - 1  # -1 to change to 0-based indexing
            group.append((factor, species))  
    model.observable_groups[values[1]] = group
