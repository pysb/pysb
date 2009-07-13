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
            site_code = ', '.join([self.format_monomer_site(m, s) for s in m.sites])
            self.__content += "  %s(%s)\n" % (m.name, site_code)
        self.__content += "end molecule types\n\n"

    def generate_reaction_rules(self):
        self.__content += "begin reaction rules\n\n"
        for r in self.model.rules:
            products_code  = ' + '.join([self.format_monomerpattern(mp) for mp in r.products])
            reactants_code = ' + '.join([self.format_monomerpattern(mp) for mp in r.reactants])
            self.__content += "  # %s\n" % (r.name)
            self.__content += "  %s -> %s    %s\n\n" % (products_code, reactants_code, r.rate.name)
        self.__content += "end reaction rules\n\n"

    def format_monomer_site(self, monomer, site):
        ret = site
        if monomer.site_states.has_key(site):
            for state in monomer.site_states[site]:
                ret += '~' + state
        return ret

    def format_monomerpattern(self, mp):
        site_pattern_code = ', '.join([])
        return '%s(%s)' % (mp.monomer.name, site_pattern_code)

