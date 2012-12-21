import pysb.core
from pysb.generator.bng import BngGenerator
import os
import subprocess
import random
import re
import itertools
import sympy
import numpy
from StringIO import StringIO


# Cached value of BNG path
_bng_path = None


def _get_bng_path():
    """Return the path to BioNetGen's BNG2.pl, based on either the BNGHOME
    environment variable if it's set, or a few hard-coded standard locations."""

    global _bng_path

    # Just return cached value if it's available
    if _bng_path:
        return _bng_path

    path_var = 'BNGHOME'
    dist_dirs = [
        '/usr/local/share/BioNetGen',
        'c:/Program Files/BioNetGen',
        ]
    # BNG 2.1.8 moved BNG2.pl up out of the Perl2 subdirectory, so to be more
    # compatible we check both the old and new locations.
    script_subdirs = ['', 'Perl2']

    def check_dist_dir(dist_dir):
        # Return the full path to BNG2.pl inside a BioNetGen distribution
        # directory, or False if directory does not contain a BNG2.pl in one of
        # the expected places.
        for subdir in script_subdirs:
            script_path = os.path.join(dist_dir, subdir, 'BNG2.pl')
            if os.access(script_path, os.F_OK):
                return script_path
        else:
            return False

    # First check the environment variable, which has the highest precedence
    if path_var in os.environ:
        script_path = check_dist_dir(os.environ[path_var])
        if not script_path:
            raise Exception('Environment variable %s is set but BNG2.pl could'
                            ' not be found there' % path_var)
    # If the environment variable isn't set, check the standard locations
    else:
        for dist_dir in dist_dirs:
            script_path = check_dist_dir(dist_dir)
            if script_path:
                break
        else:
            raise Exception('Could not find BioNetGen installed in one of the '
                            'following locations:' +
                            ''.join('\n    ' + d for d in dist_dirs))
    # Cache path for future use
    _bng_path = script_path
    return script_path

class GenerateNetworkError(RuntimeError):
    pass

generate_network_code = """
begin actions
generate_network({overwrite=>1});
end actions
"""


def _parse_bng_outfile(out_filename):
    """Parse a .gdat or .cdat outputfile produced by the ODE or SSA
       algorithm run by BNG.
       The format of the output files is an initial line of the form
       #   time   obs1    obs2    obs3  ...
       The column headers are separated by a differing number of spaces
       (not tabs). This function parses out the column names and then
       uses the numpy.loadtxt method to load the outputfile into a
       numpy record array.
    """

    try:
        out_file = open(out_filename, 'r')

        line = out_file.readline().strip() # Get the first line
        out_file.close()
        line = line[2:]  # strip off opening '# '
        raw_names = re.split('\s*', line)
        column_names = [raw_name for raw_name in raw_names if not raw_name == '']

        # Create the dtype argument for the numpy record array
        dt = zip(column_names, ('float',)*len(column_names))

        # Load the output file as a numpy record array, skip the name row
        arr = numpy.loadtxt(out_filename, dtype=dt, skiprows=1)
    
    except Exception as e:
        # FIXME special Exception/Error?
        raise Exception("problem parsing BNG outfile: " + str(e)) 
    
    return arr


def run_ssa(model, t_end=10, n_steps=100, output_dir='/tmp', cleanup=True):
    """Run a model through BNG's SSA simulator and return
    the simulation data as a numpy record array.
    """

    run_ssa_code = """
    begin actions
    generate_network({overwrite=>1});
    simulate_ssa({t_end=>%f, n_steps=>%d});\n
    end actions
    """ % (t_end, n_steps)
    
    gen = BngGenerator(model)
    bng_filename = '%s_%d_%d_temp.bngl' % (model.name,
                            os.getpid(), random.randint(0, 10000))
    gdat_filename = bng_filename.replace('.bngl', '.gdat')
    cdat_filename = bng_filename.replace('.bngl', '.cdat')
    net_filename = bng_filename.replace('.bngl', '.net')

    output = StringIO()

    try:
        #import pdb; pdb.set_trace()
        working_dir = os.getcwd()
        os.chdir(output_dir)
        bng_file = open(bng_filename, 'w')
        bng_file.write(gen.get_content())
        bng_file.write(run_ssa_code)
        bng_file.close()
        p = subprocess.Popen(['perl', _get_bng_path(), bng_filename],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (p_out, p_err) = p.communicate()
        if p.returncode:
            raise GenerateNetworkError(p_out.rstrip("at line")+"\n"+p_err.rstrip())

        output_arr = _parse_bng_outfile(gdat_filename)
        #ssa_file = open(ssa_filename, 'r')
        #output.write(ssa_file.read())
        #net_file.close()
        #if append_stdout:
        #    output.write("#\n# BioNetGen execution log follows\n# ==========")
        #    output.write(re.sub(r'(^|\n)', r'\n# ', p_out))
    finally:
        if cleanup:
            for filename in [bng_filename, gdat_filename,
                             cdat_filename, net_filename]:
                if os.access(filename, os.F_OK):
                    os.unlink(filename)
        os.chdir(working_dir)
    return output_arr

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
        p = subprocess.Popen(['perl', _get_bng_path(), bng_filename],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (p_out, p_err) = p.communicate()
        if p.returncode:
            raise GenerateNetworkError(p_out.rstrip()+"\n"+p_err.rstrip())
        net_file = open(net_filename, 'r')
        output.write(net_file.read())
        net_file.close()
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
                        bond = pysb.core.WILD
                    elif bond == '!':
                        bond = pysb.core.ANY
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
        mp = pysb.core.MonomerPattern(monomer, site_conditions, monomer_compartment)
        monomer_patterns.append(mp)

    complex_compartment = model.compartments.get(complex_compartment_name)
    cp = pysb.core.ComplexPattern(monomer_patterns, complex_compartment)
    model.species.append(cp)


def _parse_group(model, line):
    # values are number (which we ignore), name, and species list
    values = line.strip().split()
    obs = model.observables[values[1]]
    if len(values) == 3:
        # combination is a comma separated list of [coeff*]speciesnumber
        for product in values[2].split(','):
            terms = product.split('*')
            # if no coeff given (just species), insert a coeff of 1
            if len(terms) == 1:
                terms.insert(0, 1)
            obs.coefficients.append(int(terms[0]))
            # -1 to change to 0-based indexing
            obs.species.append(int(terms[1]) - 1)
