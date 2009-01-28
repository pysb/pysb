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
    name = None
    sites = []
    site_states = {}
    compartment = None

    def __init__(self, name, sites=[], site_states={}, compartment=None):
        SelfExporter.__init__(self, name)
        
        # ensure no duplicate sites
        sites_seen = {}
        for site in sites:
            sites_seen.setdefault(site, 0)
            sites_seen[site] += 1
        sites_dup = [site for site in sites_seen.keys() if sites_seen[site] > 1]
        if sites_dup:
            raise Exception("Duplicate sites specified: " + str(sites_dup))

        # ensure site_states keys are all known sites
        unknown_sites = [site for site in site_states.keys() if not site in self.sites_dict]
        if unknown_sites:
            raise Exception("Unknown sites in site_states: " + str(unknown_sites))
        # ensure site_states values are all strings
        invalid_sites = [site for (site, states) in site_states.items() if not all([type(s) == str for s in states])]
        if invalid_sites:
            raise Exception("Non-string state values in site_states for sites: " + str(invalid_sites))

        # ensure compartment is a Compartment
        if compartment and not isinstance(compartment, Compartment):
            raise Exception("compartment is not a Compartment object")

        self.sites = sites
        self.sites_dict = dict.fromkeys(sites)
        self.site_states = site_states
        self.compartment = compartment

    def __call__(self, **site_conditions):
        """Build a pattern object with convenient kwargs for the sites"""
        compartment = site_conditions.pop('compartment', self.compartment)
        return MonomerPattern(self, site_conditions, compartment)

    def __str__(self):
        return self.name + '(' + ', '.join(self.sites) + ')'



class MonomerPattern:
    monomer = None
    site_conditions = {}
    compartment = None

    def __init__(self, monomer, site_conditions, compartment):
        # ensure all keys in site_conditions are sites in monomer
        unknown_sites = [site for site in site_conditions.keys() if site not in monomer.sites_dict]
        if unknown_sites:
            raise Exception("Unknown sites in " + str(monomer) + ": " + str(unknown_sites))

        # ensure each value is None, integer, Monomer, or list of Monomers
        invalid_sites = []
        for (site, state) in site_conditions.items():
            # convert singleton monomer to list
            if isinstance(state, Monomer):
                state = [state]
                site_conditions[site] = state
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

        # ensure compartment is a Compartment
        if compartment and not isinstance(compartment, Compartment):
            raise Exception("compartment is not a Compartment object")

        self.monomer = monomer
        self.site_conditions = site_conditions
        self.compartment = compartment

    def __str__(self):
        return self.monomer.name + '(' + ', '.join([k + '=' + str(self.site_conditions[k])
                                                    for k in self.site_conditions.keys()]) + ')'



class Parameter(SelfExporter):
    value = float('nan')

    def __init__(self, name, value=float('nan')):
        SelfExporter.__init__(self, name)
        self.value = value



class Compartment(SelfExporter):
    dimension = float('nan')
    size = float('nan')
    neighbors = []

    # FIXME: sane defaults?
    def __init__(self, name, neighbors=[], dimension=3, size=1):
        SelfExporter.__init__(self, name)

        if not all([isinstance(n, Compartment) for n in neighbors]):
            raise Exception("neighbors must all be Compartments")

        self.neighbors = neighbors
        self.dimension = dimension
        self.size = size



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
