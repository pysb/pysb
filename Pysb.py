import sys
import warnings



# FIXME: make this behavior toggleable
class SelfExporter(object):
    """Expects a constructor paramter 'name', under which this object is
    inserted into the __main__ namespace."""

    do_self_export = True
    default_model = None

    def __init__(self, name):
        self.name = name

        if SelfExporter.do_self_export:
            # FIXME if name already used, add_component will succeed since it's done first.
            #   this whole thing needs to be rethought, really.
            if isinstance(self, Model):
                if SelfExporter.default_model != None:
                    raise Exception("Only one instance of Model may be declared ('%s' previously declared)" % SelfExporter.default_model.name)
                SelfExporter.default_model = self
            elif isinstance(self, (Monomer, Compartment, Parameter, Rule)):
                if SelfExporter.default_model == None:
                    raise Exception("A Model must be declared before declaring any model components")
                SelfExporter.default_model.add_component(self)

            # load self into caller's global namespace under self.name
            import inspect
            cur_module = inspect.getmodule(inspect.currentframe())
            caller_frame = inspect.currentframe()
            # iterate up through the stack until we hit a different module
            while inspect.getmodule(caller_frame) == cur_module:
                caller_frame = caller_frame.f_back
            if caller_frame.f_globals.has_key(name):
                warnings.warn("'%s' already defined" % (name))
            caller_frame.f_globals[name] = self



class Model(SelfExporter):
    def __init__(self, name):
        SelfExporter.__init__(self, name)
        self.monomers = []
        self.compartments = []
        self.parameters = []
        self.rules = []
        self.observables = []

    def add_component(self, other):
        if isinstance(other, Monomer):
            self.monomers.append(other)
        elif isinstance(other, Compartment):
            self.compartments.append(other)
        elif isinstance(other, Parameter):
            self.parameters.append(other)
        elif isinstance(other, Rule):
            self.rules.append(other)
        else:
            raise Exception("Tried to add component of unknown type (%s) to model" % type(other))

    # FIXME should this be named add_observable??
    def observe(self, name, reaction_pattern):
        try:
            reaction_pattern = as_reaction_pattern(reaction_pattern)
        except InvalidReactionPatternException as e:
            raise type(e)("Observable does not look like a reaction pattern")
        self.observables.append( (name, reaction_pattern) )

    def __repr__(self):
        return "%s( \\\n    monomers=%s \\\n    compartments=%s\\\n    parameters=%s\\\n    rules=%s\\\n)" % \
            (self.__class__.__name__, repr(self.monomers), repr(self.compartments), repr(self.parameters), repr(self.rules))



class Monomer(SelfExporter):
    def __init__(self, name, sites=[], site_states={}, compartment=None):
        SelfExporter.__init__(self, name)

        # convert single site string to list
        if type(sites) == str:
            sites = [sites]
        
        # ensure no duplicate sites
        sites_seen = {}
        for site in sites:
            sites_seen.setdefault(site, 0)
            sites_seen[site] += 1
        sites_dup = [site for site in sites_seen.keys() if sites_seen[site] > 1]
        if sites_dup:
            raise Exception("Duplicate sites specified: " + str(sites_dup))

        # ensure site_states keys are all known sites
        unknown_sites = [site for site in site_states.keys() if not site in sites_seen]
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

    def __repr__(self):
        return  '%s(name=%s, sites=%s, site_states=%s, compartment=%s)' % \
            (self.__class__.__name__, repr(self.name), repr(self.sites), repr(self.site_states), self.compartment and self.compartment.name or None)



class MonomerAny(Monomer):
    def __init__(self):
        # don't call Monomer.__init__ since this doesn't want
        # SelfExporter stuff and has no user-accessible API
        self.name = 'ANY'
        self.sites = None
        self.sites_dict = {}
        self.site_states = {}
        self.compartment = None

    def __repr__(self):
        return self.name



class MonomerWild(Monomer):
    def __init__(self):
        # don't call Monomer.__init__ since this doesn't want
        # SelfExporter stuff and has no user-accessible API
        self.name = 'WILD'
        self.sites = None
        self.sites_dict = {}
        self.site_states = {}
        self.compartment = None

    def __repr__(self):
        return self.name



class MonomerPattern(object):
    def __init__(self, monomer, site_conditions, compartment):
        # ensure all keys in site_conditions are sites in monomer
        unknown_sites = [site for site in site_conditions.keys() if site not in monomer.sites_dict]
        if unknown_sites:
            raise Exception("Unknown sites in " + str(monomer) + ": " + str(unknown_sites))

        # ensure each value is None, integer, string, (string,integer), (string,WILD), Monomer, or list of Monomers
        # FIXME: support state sites
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
            elif type(state) == str:
                continue
            elif type(state) == tuple and type(state[0]) == str and (type(state[1]) == int or state[1] == WILD):
                continue
            elif type(state) == list and all([isinstance(s, Monomer) for s in state]):
                continue
            invalid_sites.append(site)
        if invalid_sites:
            raise Exception("Invalid state value for sites: " + '; '.join(['%s=%s' % (s,str(site_conditions[s])) for s in invalid_sites]))

        # ensure compartment is a Compartment
        if compartment and not isinstance(compartment, Compartment):
            raise Exception("compartment is not a Compartment object")

        self.monomer = monomer
        self.site_conditions = site_conditions
        self.compartment = compartment

    def __add__(self, other):
        if isinstance(other, MonomerPattern):
            return ReactionPattern([ComplexPattern([self]), ComplexPattern([other])])
        if isinstance(other, ComplexPattern):
            return ReactionPattern([ComplexPattern([self]), other])
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, MonomerPattern):
            return ComplexPattern([self, other])
        else:
            return NotImplemented

    def __rshift__(self, other):
        if isinstance(other, (MonomerPattern, ComplexPattern, ReactionPattern)):
            return (self, other, False)
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, (MonomerPattern, ComplexPattern, ReactionPattern)):
            return (self, other, True)
        else:
            return NotImplemented

    def __repr__(self):
        return self.monomer.name + '(' + ', '.join([k + '=' + str(self.site_conditions[k])
                                                    for k in self.monomer.sites
                                                    if self.site_conditions.has_key(k)]) + ')'



class ComplexPattern(object):
    """Represents a bound set of MonomerPatterns, i.e. a complex.  In
    BNG terms, a list of patterns combined with the '.' operator)."""

    def __init__(self, monomer_patterns):
        self.monomer_patterns = monomer_patterns

    def __add__(self, other):
        if isinstance(other, ComplexPattern):
            return ReactionPattern([self, other])
        elif isinstance(other, MonomerPattern):
            return ReactionPattern([self, ComplexPattern([other])])
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, MonomerPattern):
            return ComplexPattern(self.monomer_patterns + [other])
        else:
            return NotImplemented

    def __rshift__(self, other):
        if isinstance(other, (MonomerPattern, ComplexPattern, ReactionPattern)):
            return (self, other, False)
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, (MonomerPattern, ComplexPattern, ReactionPattern)):
            return (self, other, True)
        else:
            return NotImplemented

    def __repr__(self):
        return ' * '.join([repr(p) for p in self.monomer_patterns])



class ReactionPattern(object):
    """Represents a complete pattern for the product or reactant side
    of a rule.  Essentially a thin wrapper around a list of
    ComplexPatterns."""

    def __init__(self, complex_patterns):
        self.complex_patterns = complex_patterns

    def __add__(self, other):
        if isinstance(other, MonomerPattern):
            return ReactionPattern(self.complex_patterns + [ComplexPattern([other])])
        elif isinstance(other, ComplexPattern):
            return ReactionPattern(self.complex_patterns + [other])
        else:
            return NotImplemented

    def __rshift__(self, other):
        """Irreversible reaction"""
        if isinstance(other, (MonomerPattern, ComplexPattern, ReactionPattern)):
            return (self, other, False)
        else:
            return NotImplemented

    def __ne__(self, other):
        """Reversible reaction"""
        if isinstance(other, (MonomerPattern, ComplexPattern, ReactionPattern)):
            return (self, other, True)
        else:
            return NotImplemented

    def __repr__(self):
        return ' + '.join([repr(p) for p in self.complex_patterns])



def as_reaction_pattern(v):
    """Internal helper to 'upgrade' a Complex- or MonomerPattern to a
    complete ReactionPattern."""
    if isinstance(v, ReactionPattern):
        return v
    elif isinstance(v, ComplexPattern):
        return ReactionPattern([v])
    elif isinstance(v, MonomerPattern):
        return ReactionPattern([ComplexPattern([v])])
    else:
        raise InvalidReactionPatternException



class Parameter(SelfExporter):
    def __init__(self, name, value=float('nan')):
        SelfExporter.__init__(self, name)
        self.value = value

    def __repr__(self):
        return  '%s(name=%s, value=%s)' % (self.__class__.__name__, repr(self.name), repr(self.value))



class Compartment(SelfExporter):
    # FIXME: sane defaults?
    def __init__(self, name, neighbors=[], dimension=3, size=1):
        SelfExporter.__init__(self, name)

        if not all([isinstance(n, Compartment) for n in neighbors]):
            raise Exception("neighbors must all be Compartments")

        self.neighbors = neighbors
        self.dimension = dimension
        self.size = size

    def __repr__(self):
        return  '%s(name=%s, dimension=%s, size=%s, neighbors=%s)' % \
            (self.__class__.__name__, repr(self.name), repr(self.dimension), repr(self.size), repr(self.neighbors))
        # FIXME don't recurse into neighbors, just print their names



class Rule(SelfExporter):
    def __init__(self, name, reaction_pattern_set, rate_forward, rate_reverse=None):
        SelfExporter.__init__(self, name)

        # FIXME: This tuple thing is ugly (used to support >> and <> operators between ReactionPatterns).
        # This is how the reactant and product ReactionPatterns are passed, along with is_reversible.
        if not isinstance(reaction_pattern_set, tuple) and len(reaction_pattern_set) != 3:
            raise Exception("reaction_pattern_set must be a tuple of (ReactionPattern, ReactionPattern, Boolean)")

        try:
            reactant_pattern = as_reaction_pattern(reaction_pattern_set[0])
        except InvalidReactionPatternException as e:
            raise type(e)("Reactant does not look like a reaction pattern")

        try:
            product_pattern = as_reaction_pattern(reaction_pattern_set[1])
        except InvalidReactionPatternException as e:
            raise type(e)("Product does not look like a reaction pattern")

        self.is_reversible = reaction_pattern_set[2]

        if not isinstance(rate_forward, Parameter):
            raise Exception("Forward rate must be a Parameter")
        if self.is_reversible and not isinstance(rate_reverse, Parameter):
            raise Exception("Reverse rate must be a Parameter")

        self.reactant_pattern = reactant_pattern
        self.product_pattern = product_pattern
        self.rate_forward = rate_forward
        self.rate_reverse = rate_reverse
        # TODO: ensure all numbered sites are referenced exactly twice within each of reactants and products

    def __repr__(self):
        ret = '%s(name=%s, reactants=%s, products=%s, rate_forward=%s' % \
            (self.__class__.__name__, repr(self.name), repr(self.reactant_pattern), repr(self.product_pattern), repr(self.rate_forward))
        if self.is_reversible:
            ret += ', rate_reverse=%s' % repr(self.rate_reverse)
        ret += ')'
        return ret



class InvalidReactionPatternException(Exception):
    pass



ANY = MonomerAny()
WILD = MonomerWild()
