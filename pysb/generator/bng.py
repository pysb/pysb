import pysb


class BngGenerator(object):

    def __init__(self, model):
        self.model = model
        self.__content = None

    def get_content(self):
        if self.__content == None:
            self.generate_content()
        return self.__content

    def generate_content(self):
        self.__content = ''
        self.generate_parameters()
        self.generate_compartments()
        self.generate_molecule_types()
        self.generate_reaction_rules()
        self.generate_observables()
        self.generate_species()

    def generate_parameters(self):
        self.__content += "begin parameters\n"
        max_length = max(len(p.name) for p in self.model.parameters)
        for p in self.model.parameters:
            self.__content += ("  %-" + str(max_length) + "s   %e\n") % (p.name, p.value)
        self.__content += "end parameters\n\n"

    def generate_compartments(self):
        if not self.model.compartments:
            return
        self.__content += "begin compartments\n"
        for c in self.model.compartments:
            if c.parent is None:
                parent_name = ''
            else:
                parent_name = c.parent.name
            if c.size is None:
                size = "1.0"
            else:
                size = c.size.name
            self.__content += ("  %s  %d  %s  %s\n") % (c.name, c.dimension, size, parent_name)
        self.__content += "end compartments\n\n"        

    def generate_molecule_types(self):
        self.__content += "begin molecule types\n"
        for m in self.model.monomers:
            site_code = ','.join([format_monomer_site(m, s) for s in m.sites])
            self.__content += "  %s(%s)\n" % (m.name, site_code)
        self.__content += "end molecule types\n\n"

    def generate_reaction_rules(self):
        if not self.model.rules:
            return
        self.__content += "begin reaction rules\n"
        max_length = max(len(r.name) for r in self.model.rules) + 1  # +1 for the colon
        for r in self.model.rules:
            label = r.name + ':'
            reactants_code = format_reactionpattern(r.reactant_pattern)
            products_code  = format_reactionpattern(r.product_pattern)
            arrow = '->'
            if r.is_reversible:
                arrow = '<->'
            self.__content += ("  %-" + str(max_length) + "s  %s %s %s    %s") % \
                (label, reactants_code, arrow, products_code, r.rate_forward.name)
            if r.is_reversible:
                self.__content += ', %s' % r.rate_reverse.name
            if r.delete_molecules:
                self.__content += ' DeleteMolecules'
            self.__content += "\n"
        self.__content += "end reaction rules\n\n"

    def generate_observables(self):
        if not self.model.observable_patterns:
            return
        max_length = max(len(name) for name, pattern in self.model.observable_patterns)
        self.__content += "begin observables\n"
        for name, pattern in self.model.observable_patterns:
            observable_code = format_reactionpattern(pattern)
            self.__content += ("  Molecules %-" + str(max_length) + "s   %s\n") % (name, observable_code)
        self.__content += "end observables\n\n"

    def generate_species(self):
        if not self.model.initial_conditions:
            raise Exception("BNG generator requires initial conditions.")
        species_codes = [format_complexpattern(cp) for cp, param in self.model.initial_conditions]
        max_length = max(len(code) for code in species_codes)
        self.__content += "begin species\n"
        for i, code in enumerate(species_codes):
            param = self.model.initial_conditions[i][1]
            self.__content += ("  %-" + str(max_length) + "s   %s\n") % (code, param.name)
        self.__content += "end species\n"



def format_monomer_site(monomer, site):
    ret = site
    if monomer.site_states.has_key(site):
        for state in monomer.site_states[site]:
            ret += '~' + state
    return ret

def format_reactionpattern(rp):
    return ' + '.join([format_complexpattern(cp) for cp in rp.complex_patterns])

def format_complexpattern(cp):
    ret = '.'.join([format_monomerpattern(mp) for mp in cp.monomer_patterns])
    if cp.compartment is not None:
        ret = '@%s:%s' % (cp.compartment.name, ret)
    if cp.match_once:
        ret = '{MatchOnce}' + ret
    return ret

def format_monomerpattern(mp):
    # sort sites in the same order given in the original Monomer
    site_conditions = sorted(mp.site_conditions.items(),
                             key=lambda x: mp.monomer.sites.index(x[0]))
    site_pattern_code = ','.join([format_site_condition(site, state) for (site, state) in site_conditions])
    ret = '%s(%s)' % (mp.monomer.name, site_pattern_code)
    if mp.compartment is not None:
        ret = '%s@%s' % (ret, mp.compartment.name)
    return ret

def format_site_condition(site, state):
    # empty
    if state == None:
        state_code = ''
    # single bond
    elif type(state) == int:
        state_code = '!' + str(state)
    # multiple bonds
    elif type(state) == list and all(isinstance(s, int) for s in state):
        state_code = ''.join('!%d' % s for s in state)
    # state
    elif type(state) == str:
        state_code = '~' + state
    # state AND single bond
    elif type(state) == tuple:
        # bond is wildcard (zero or more unspecified bonds)
        if state[1] == pysb.WILD:
            state = (state[0], '?')
        state_code = '~%s!%s' % state
    # one or more unspecified bonds
    elif state == pysb.ANY:
        state_code = '!+'
    else:
        raise Exception("BNG generator has encountered an unknown element in a rule pattern site condition.")
    return '%s%s' % (site, state_code)
