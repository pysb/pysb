import Pysb


class BngGenerator(object):

    def __init__(self, model):
        self.model = model
        self.__content = None

    def __get_content(self):
        if self.__content == None:
            self.generate_content()
        return self.__content

    content = property(fget=__get_content)

    def generate_content(self):
        self.__content = ''
        self.generate_parameters()
        self.generate_molecule_types()
        self.generate_reaction_rules()
        self.generate_observables()

    def generate_parameters(self):
        self.__content += "begin parameters\n"
        max_length = max([len(p.name) for p in self.model.parameters])
        for p in self.model.parameters:
            self.__content += ("  %-" + str(max_length) + "s   %e\n") % (p.name, p.value)
        self.__content += "end parameters\n\n"

    def generate_molecule_types(self):
        self.__content += "begin molecule types\n"
        for m in self.model.monomers:
            site_code = ','.join([format_monomer_site(m, s) for s in m.sites])
            self.__content += "  %s(%s)\n" % (m.name, site_code)
        self.__content += "end molecule types\n\n"

    def generate_reaction_rules(self):
        self.__content += "begin reaction rules\n"
        max_length = max([len(r.name) for r in self.model.rules]) + 1  # +1 for the colon
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
            self.__content += "\n"
        self.__content += "end reaction rules\n\n"

    def generate_observables(self):
        max_length = max([len(name) for name, pattern in self.model.observables])
        self.__content += "begin observables\n"
        for name, pattern in self.model.observables:
            observable_code = format_reactionpattern(pattern)
            self.__content += ("  %-" + str(max_length) + "s   %s\n") % (name, observable_code)
        self.__content += "end observables\n\n"



def format_monomer_site(monomer, site):
    ret = site
    if monomer.site_states.has_key(site):
        for state in monomer.site_states[site]:
            ret += '~' + state
    return ret

def format_reactionpattern(rp):
    return ' + '.join([format_complexpattern(cp) for cp in rp.complex_patterns])

def format_complexpattern(cp):
    return '.'.join([format_monomerpattern(mp) for mp in cp.monomer_patterns])

def format_monomerpattern(mp):
    # sort sites in the same order given in the original Monomer
    site_conditions = sorted(mp.site_conditions.items(),
                             key=lambda x: mp.monomer.sites.index(x[0]))
    site_pattern_code = ','.join([format_site_condition(site, state) for (site, state) in site_conditions])
    return '%s(%s)' % (mp.monomer.name, site_pattern_code)

def format_site_condition(site, state):
    if state == None:
        state_code = ''
    elif type(state) == int:
        state_code = '!' + str(state)
    elif type(state) == str:
        state_code = '~' + state
    elif type(state) == tuple:
        if state[1] == Pysb.WILD:
            state = (state[0], '?')
        state_code = '~%s!%s' % state
    elif type(state) == list:
        if len(state) == 1:
            if (state[0] == Pysb.ANY):
                state_code = '!+'
            else:
                raise Exception("BNG generator does not support named monomers in rule pattern site conditions.")
        else:
            raise Exception("BNG generator does not support multi-monomer lists in rule pattern site conditions.")
    else:
        raise Exception("BNG generator has encountered an unknown element in a rule pattern site condition.")
    return '%s%s' % (site, state_code)
