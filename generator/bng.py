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

    def generate_parameters(self):
        self.__content += "begin parameters\n"
        max_length = max([len(p.name) for p in self.model.parameters])
        for p in self.model.parameters:
            self.__content += ("  %-" + str(max_length) + "s   %e\n") % (p.name, p.value)
        self.__content += "end parameters\n\n"

    def generate_molecule_types(self):
        self.__content += "begin molecule types\n"
        for m in self.model.monomers:
            site_code = ', '.join([format_monomer_site(m, s) for s in m.sites])
            self.__content += "  %s(%s)\n" % (m.name, site_code)
        self.__content += "end molecule types\n\n"

    def generate_reaction_rules(self):
        self.__content += "begin reaction rules\n\n"
        for r in self.model.rules:
            reactants_code = ' + '.join([format_monomerpattern(mp) for mp in r.reactants])
            products_code  = ' + '.join([format_monomerpattern(mp) for mp in r.products])
            self.__content += "  # %s\n" % (r.name)
            self.__content += "  %s -> %s    %s\n\n" % (reactants_code, products_code, r.rate.name)
        self.__content += "end reaction rules\n\n"



def format_monomer_site(monomer, site):
    ret = site
    if monomer.site_states.has_key(site):
        for state in monomer.site_states[site]:
            ret += '~' + state
    return ret

def format_monomerpattern(mp):
    # sort sites in the same order given in the original Monomer
    site_conditions = sorted(mp.site_conditions.items(),
                             key=lambda x: mp.monomer.sites.index(x[0]))
    site_pattern_code = ', '.join([format_site_condition(site, state) for (site, state) in site_conditions])
    return '%s(%s)' % (mp.monomer.name, site_pattern_code)

def format_site_condition(site, state):
    state_code = ''
    if state == None:
        pass
    elif type(state) == int:
        state_code += '!' + str(state)
    elif type(state) == list:
        if len(state) == 1:
            if (state[0] == Pysb.ANY):
                state_code += '!+'
            else:
                state_code += '!' + state[0].name
        else:
            raise Exception("BNG generator does not support multi-species lists in rule pattern site conditions.")
    return '%s%s' % (site, state_code)
