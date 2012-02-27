import pysb

class KappaGenerator(object):

    # Dialect can be either 'complx' or 'kasim' (default)
    def __init__(self, model, dialect='kasim'):
        self.model = model
        self.__content = None
        self.dialect = dialect

    def get_content(self):
        if self.__content == None:
            self.generate_content()
        return self.__content

    def generate_content(self):
        self.__content = ''
        #self.generate_parameters()
        #self.generate_compartments()

        # Agent declarations appear to be required in kasim
        # but prohibited in complx
        if (self.dialect == 'kasim'):
            self.generate_molecule_types() 

        self.generate_reaction_rules()
        self.generate_observables()
        self.generate_species()

    #def generate_parameters(self):
    #    self.__content += "begin parameters\n"
    #    max_length = max(len(p.name) for p in self.model.parameters)
    #    for p in self.model.parameters:
    #        self.__content += ("  %-" + str(max_length) + "s   %e\n") % (p.name, p.value)
    #    self.__content += "end parameters\n\n"

    #def generate_compartments(self):
    #    self.__content += "begin compartments\n"
    #    for c in self.model.compartments:
    #        if c.parent is None:
    #            parent_name = ''
    #        else:
    #            parent_name = c.parent.name
    #        self.__content += ("  %s  %d  %f  %s\n") % (c.name, c.dimension, c.size, parent_name)
    #    self.__content += "end compartments\n\n"        

    def generate_molecule_types(self):
        for m in self.model.monomers:
            site_code = ','.join([format_monomer_site(m, s) for s in m.sites])
            self.__content += "%%agent: %s(%s)\n" % (m.name, site_code)
        self.__content += "\n"

    def generate_reaction_rules(self):
        #self.__content += "begin reaction rules\n"
        #max_length = max(len(r.name) for r in self.model.rules) + 1  # +1 for the colon
        max_length = 0
        for r in self.model.rules:
            label = '\'' + r.name + '\''
            reactants_code = format_reactionpattern(r.reactant_pattern)
            products_code  = format_reactionpattern(r.product_pattern)
            arrow = '->'
            self.__content += ("%s %s %s %s @ %s") % \
                (label, reactants_code, arrow, products_code, r.rate_forward.value)
            self.__content += "\n"

            if r.is_reversible:
              label = '\'' + r.name + '_rev' + '\''
              self.__content += ("%s %s %s %s @ %s") % \
                (label, products_code, arrow, reactants_code, r.rate_reverse.value)
              self.__content += "\n"

        self.__content += "\n"

    def generate_observables(self):
        if not self.model.observable_patterns:
            return
        #max_length = max(len(name) for name, pattern in self.model.observable_patterns)
        max_length = 0
        #self.__content += "begin observables\n"
        for name, pattern in self.model.observable_patterns:
            name = '\'' + name + '\''
            observable_code = format_reactionpattern(pattern)
            self.__content += ("%%obs: %s %s\n") % (name, observable_code)
        self.__content += "\n"
        #self.__content += "end observables\n\n"

    def generate_species(self):
        if not self.model.initial_conditions:
            raise Exception("BNG generator requires initial conditions.")
        species_codes = [format_complexpattern(cp) for cp, param in self.model.initial_conditions]
        #max_length = max(len(code) for code in species_codes)
        max_length = 0
        #self.__content += "begin species\n"
        for i, code in enumerate(species_codes):
            param = self.model.initial_conditions[i][1]
            #self.__content += ("%%init:  %-" + str(max_length) + "s   %s\n") % (code, param.name)
            if (self.dialect == 'kasim'):
                # Switched from %g (float) to %d (int) because kappa didn't like scientific notation
                # for large integers
                self.__content += ("%%init: %d %s\n") % (param.value, code)
                #self.__content += ("%%init: %10g %s\n") % (param.value, code)
            else:
                # Switched from %g (float) to %d (int) because kappa didn't like scientific notation
                # for large integers
                self.__content += ("%%init: %10d * %s\n") % (param.value, code)
                #self.__content += ("%%init: %10g * %s\n") % (param.value, code)

        self.__content += "\n"


def format_monomer_site(monomer, site):
    ret = site
    if monomer.site_states.has_key(site):
        for state in monomer.site_states[site]:
            ret += '~' + state
    return ret

def format_reactionpattern(rp):
    return ','.join([format_complexpattern(cp) for cp in rp.complex_patterns])

def format_complexpattern(cp):
    ret = ','.join([format_monomerpattern(mp) for mp in cp.monomer_patterns])
    if cp.compartment is not None:
        ret = '@%s:%s' % (cp.compartment.name, ret)
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
    if state == None:
        state_code = ''
    elif type(state) == int:
        state_code = '!' + str(state)
    elif type(state) == list:
        raise Exception("Kappa generator does not support multiple bonds to a single site.")
    elif type(state) == str:
        state_code = '~' + state
    elif type(state) == tuple:
        if state[1] == pysb.WILD:
            state = (state[0], '?')
        state_code = '~%s!%s' % state
    elif state == pysb.ANY:
        state_code = '!_'
    else:
        raise Exception("Kappa generator has encountered an unknown element in a rule pattern site condition.")
    return '%s%s' % (site, state_code)
