import pysb
from pysb.generator.bng import BngGenerator
import os
import subprocess
import random
import re
import sympy
from StringIO import StringIO

# not ideal, but it will work for now during development
pkg_path = None
pkg_paths_to_check = [
    '/usr/local/share/BioNetGen',
    'c:/Program Files/BioNetGen'
    ]
for check_path in pkg_paths_to_check:
    for subdir in ('', 'Perl2'):
        path = os.path.join(check_path, subdir)
        if os.access(os.path.join(path, 'BNG2.pl'), os.F_OK):
            pkg_path = path
            break
if pkg_path is None:
    msg = "Could not find BioNetGen installed in one of the following locations:\n    " + \
        '\n    '.join(pkg_paths_to_check)
    raise Exception(msg)

class GenerateNetworkError(RuntimeError):
    pass

generate_network_code = """
begin actions
generate_network({overwrite=>1});
end actions
"""


def generate_network(model, cleanup=True, append_stdout=False):
    """Run a model through BNG's generate_network function and return
    the content of the resulting .net file as a string"""
    gen = BngGenerator(model)
    bng_filename = '%s_%d_%d_temp.bngl' % (model.name, os.getpid(), random.randint(0, 10000))
    net_filename = bng_filename.replace('.bngl', '.net')
    output = StringIO()
    try:
        bng_file = open(bng_filename, 'w')
        bng_file.write(gen.get_content())
        bng_file.write(generate_network_code)
        bng_file.close()
        p = subprocess.Popen(['perl', pkg_path + 'BNG2.pl', bng_filename],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (p_out, p_err) = p.communicate()
        if p.returncode:
            raise GenerateNetworkError(p_err.rstrip())
        net_file = open(net_filename, 'r')
        output.write(net_file.read())
        if append_stdout:
            output.write("#\n# BioNetGen execution log follows\n# ==========")
            output.write(re.sub(r'(^|\n)', r'\n# ', p_out))
    finally:
        if cleanup:
            for filename in [bng_filename, net_filename]:
                if os.access(filename, os.F_OK):
                    os.unlink(filename)
    return output.getvalue()


def generate_equations(model):
    # only need to do this once
    # TODO track "dirty" state, i.e. "has model been modified?"
    #   or, use a separate "math model" object to contain ODEs
    if model.odes:
        return
    lines = iter(generate_network(model).split('\n'))
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
        reaction_cache = {}
        while True:
            line = lines.next()
            if 'end reactions' in line: break
            (number, reactants, products, rate, rule) = line.strip().split()
            # the -1 is to switch from one-based to zero-based indexing
            reactants = tuple(int(r) - 1 for r in reactants.split(','))
            products = tuple(int(p) - 1 for p in products.split(','))
            rate = rate.rsplit('*')
            (rule_name, is_reverse) = re.match(r'#(\w+)(?:\((reverse)\))?', rule).groups()
            is_reverse = bool(is_reverse)
            r_names = ['s%d' % r for r in reactants]
            combined_rate = sympy.Mul(*[sympy.S(t) for t in r_names + rate]) 
            rule = model.rules[rule_name]
            reaction = {
                'reactants': reactants,
                'products': products,
                'rate': combined_rate,
                'rule': rule_name,
                'reverse': is_reverse,
                }
            model.reactions.append(reaction)
            key = (rule_name, reactants, products)
            key_reverse = (rule_name, products, reactants)
            reaction_bd = reaction_cache.get(key_reverse)
            if reaction_bd is None:
                # make a copy of the reaction dict
                reaction_bd = dict(reaction)
                # default to false until we find a matching reverse reaction
                reaction_bd['reversible'] = False
                reaction_cache[key] = reaction_bd
                model.reactions_bidirectional.append(reaction_bd)
            else:
                reaction_bd['reversible'] = True
                reaction_bd['rate'] -= combined_rate
            for p in products:
                model.odes[p] += combined_rate
            for r in reactants:
                model.odes[r] -= combined_rate
        # fix up reactions whose reverse version we saw first
        for r in model.reactions_bidirectional:
            if r['reverse']:
                r['reactants'], r['products'] = r['products'], r['reactants']
                r['rate'] *= -1
            # now the 'reverse' value is no longer needed
            del r['reverse']

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
    index, species, value = line.strip().split()
    complex_compartment_name, complex_string = re.match(r'(?:@(\w+)::)?(.*)', species).groups()
    monomer_strings = complex_string.split('.')
    monomer_patterns = []
    for ms in monomer_strings:
        monomer_name, site_strings, monomer_compartment_name = re.match(r'(\w+)\(([^)]*)\)(?:@(\w+))?', ms).groups()
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
        monomer_compartment = model.compartments.get(monomer_compartment_name)
        mp = pysb.MonomerPattern(monomer, site_conditions, monomer_compartment)
        monomer_patterns.append(mp)

    complex_compartment = model.compartments.get(complex_compartment_name)
    cp = pysb.ComplexPattern(monomer_patterns, complex_compartment)
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
