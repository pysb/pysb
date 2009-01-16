import sys



class SelfExporter:
    """Expects a constructor paramter 'name', under which this object is
    inserted into the __main__ namespace."""

    name = None

    def __init__(self, name):
        self.name = name
        # load self into __main__ namespace
        main = sys.modules['__main__']
        if hasattr(main, name):
            raise Exception("'%s' already defined" % (name))
        setattr(main, name, self)



class Monomer(SelfExporter):
    name = '**UNNAMED**'
    sites = []

    def __init__(self, name, sites):
        SelfExporter.__init__(self, name)
        
        # ensure no duplicate sites
        sites_seen = {}
        for site in sites:
            sites_seen.setdefault(site, 0)
            sites_seen[site] += 1
        sites_dup = [site for site in sites_seen.keys() if sites_seen[site] > 1]
        if sites_dup:
            raise Exception("Duplicate sites specified: " + str(sites_dup))

        self.sites = sites
        self.sites_dict = dict.fromkeys(sites)

    def __call__(self, **site_states):
        """Build a pattern object with convenient kwargs for the sites"""
        return MonomerPattern(self, site_states)

    def __str__(self):
        return self.name + '(' + ', '.join(self.sites) + ')'



class MonomerPattern:
    monomer = None
    site_states = {}

    def __init__(self, monomer, site_states):
        # ensure all keys in site_states are sites in monomer
        unknown_sites = [site for site in site_states.keys() if site not in monomer.sites_dict]
        if unknown_sites:
            raise Exception("Unknown sites in " + str(monomer) + ": " + str(unknown_sites))

        # ensure each value is None, integer, Monomer, or list of Monomers
        invalid_sites = []
        for (site, state) in site_states.items():
            # convert singleton monomer to list
            if isinstance(state, Monomer):
                state = [state]
                site_states[site] = state
            # pass through to next iteration if state type is ok
            if state == None:
                continue
            elif type(state) == int:
                continue
            elif type(state) == list and all([isinstance(s, Monomer) for s in state]):
                continue
            invalid_sites.append(site)
        if invalid_sites:
            raise Exception("Invalid state value for sites: " + str(invalid_sites))

        self.monomer = monomer
        self.site_states = site_states

    def __str__(self):
        return self.monomer.name + '(' + ', '.join([k + '=' + str(self.site_states[k]) for k in self.site_states.keys()]) + ')'



class Parameter(SelfExporter):
    value = float('nan')

    def __init__(self, name, value=float('nan')):
        SelfExporter.__init__(self, name)
        self.value = value



class Rule(SelfExporter):
    reactants = []
    products = []
    rate = []

    def __init__(self, name, reactants, products, rate):
        SelfExporter.__init__(self, name)

        if not all([isinstance(r, MonomerPattern) for r in reactants]):
            raise Exception("Reactants must all be MonomerPatterns")
        if not all([isinstance(p, MonomerPattern) for p in products]):
            raise Exception("Products must all be MonomerPatterns")
        if not isinstance(rate, Parameter):
            raise Exception("Rate must be a Parameter")

        self.reactants = reactants
        self.products = products
        self.rate = rate
        # TODO: ensure all numbered sites are referenced exactly twice within each of reactants and products
