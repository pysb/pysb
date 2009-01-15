class Monomer:
    name = '**UNNAMED**'
    sites = []

    def __init__(self, name, sites):
        self.name = name
        # TODO: ensure no duplicate sites
        self.sites = sites

    def m(self, **site_states):
        return MonomerPattern(self, site_states)

    def __str__(self):
        return self.name + '(' + ', '.join(self.sites) + ')'



class MonomerPattern:
    monomer = None
    site_states = {}

    def __init__(self, monomer, site_states):
        self.monomer = monomer
        # TODO: ensure all keys in site_states are sites in monomer
        # TODO: ensure each value is a monomer, lists of monomers, integer, or None
        self.site_states = site_states

    def __str__(self):
        return self.monomer.name + '(' + ', '.join([k + '=' + str(self.site_states[k]) for k in self.site_states.keys()]) + ')'



class Rule:
    name = '**UNNAMED**'
    reactants = []
    products = []
    rate = []

    def __init__(self, name, reactants, products, rate):
        self.name = name
        self.reactants = reactants
        self.products = products
        self.rate = rate
        # TODO: ensure all numbered sites are referenced exactly twice within each of reactants and products
