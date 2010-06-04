import sys
import warnings
import logging

#set core logger to DEBUG level
logging.basicConfig()
clogger = logging.getLogger("CoreFile")
clogger.setLevel(logging.DEBUG)

clogger.info("INITIALIZING")

def Observe(*args):
    return SelfExporter.default_model.observe(*args)

def Initial(*args):
    return SelfExporter.default_model.initial(*args)


# FIXME: make this behavior toggleable
class SelfExporter(object):
    """Expects a constructor paramter 'name', under which this object is
    inserted into the namespace from which the Model constructor was called."""

    clogger.debug('in SelfExporter')

    do_self_export = True
    default_model = None
    target_globals = None   # the globals dict to which we'll export our symbols

    def __init__(self, name, __export=True):
        self.name = name

        if SelfExporter.do_self_export and __export:
            # FIXME if name already used, add_component will succeed since it's done first.
            #   this whole thing needs to be rethought, really.
            if isinstance(self, Model):
                if SelfExporter.default_model is not None:
                    raise Exception("Only one instance of Model may be declared ('%s' previously declared)" % SelfExporter.default_model.name)
                # determine the module from which the Model constructor was called
                import inspect
                cur_module = inspect.getmodule(inspect.currentframe())
                caller_frame = inspect.currentframe()
                # iterate up through the stack until we hit a different module
                while inspect.getmodule(caller_frame) == cur_module:
                    caller_frame = caller_frame.f_back
                SelfExporter.target_globals = caller_frame.f_globals
                SelfExporter.default_model = self
            elif isinstance(self, (Monomer, Compartment, Parameter, Rule)):
                if SelfExporter.default_model == None:
                    raise Exception("A Model must be declared before declaring any model components")
                SelfExporter.default_model.add_component(self)

            # load self into target namespace under self.name
            if SelfExporter.target_globals.has_key(name):
                warnings.warn("'%s' already defined" % (name))
            SelfExporter.target_globals[name] = self



class Model(SelfExporter):

    clogger.debug('in Model')

    def __init__(self, name='model', __export=True):
        SelfExporter.__init__(self, name, __export)
        self.monomers = []
        self.compartments = []
        self.parameters = []
        self.parameter_overrides = {}
        self.rules = []
        self.species = []
        self.odes = []
        self.observable_patterns = []
        self.observable_groups = {}  # values are tuples of factor,speciesnumber
        self.initial_conditions = []

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
            raise type(e)("Observable pattern does not look like a ReactionPattern")
        self.observable_patterns.append( (name, reaction_pattern) )

    def initial(self, complex_pattern, value):
        try:
            complex_pattern = as_complex_pattern(complex_pattern)
        except InvalidComplexPatternException as e:
            raise type(e)("Initial condition species does not look like a ComplexPattern")
        if not isinstance(value, Parameter):
            raise Exception("Value must be a Parameter")
        if not complex_pattern.is_concrete():
            raise Exception("Pattern must be concrete (all sites specified)")
        self.initial_conditions.append( (complex_pattern, value) )

    def parameter(self, name):
        # FIXME rename to get_parameter
        # FIXME probably want to store params in a dict by name instead of a list
        try:
            return (p for p in self.parameters if p.name == name).next()
        except StopIteration:
            return None

    def set_parameter(self, name, value):
        """Overrides the baseline value of a parameter."""
        if not self.parameter(name):
            raise Exception("Model does not have a parameter named '%s'" % name)
        self.parameter_overrides[name] = Parameter(name, value, False)

    def reset_parameters(self):
        """Resets all parameters back to the baseline defined in the model script."""
        self.parameter_overrides = {}

    def get_monomer(self, name):
        # FIXME probably want to store monomers in a dict by name instead of a list
        try:
            return (m for m in self.monomers if m.name == name).next()
        except StopIteration:
            return None

    def get_species_index(self, complex_pattern):
        # FIXME I don't even want to think about the inefficiency of this, but at least it works
        try:
            return (i for i, s_cp in enumerate(self.species) if s_cp.is_equivalent_to(complex_pattern)).next()
        except StopIteration:
            return None

    def __repr__(self):
        return "%s( \\\n    monomers=%s \\\n    compartments=%s\\\n    parameters=%s\\\n    rules=%s\\\n)" % \
            (self.__class__.__name__, repr(self.monomers), repr(self.compartments), repr(self.parameters), repr(self.rules))



class Monomer(SelfExporter):
    """The Monomer class creates monomers with the specified sites, state-sites, and compartment
    """

    clogger.debug('in Monomer')

    def __init__(self, name, sites=[], site_states={}, compartment=None, __export=True):
        SelfExporter.__init__(self, name, __export)

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

    def __call__(self, *dict_site_conditions, **named_site_conditions):
        """Build a pattern object with convenient kwargs for the sites"""
        site_conditions = named_site_conditions.copy()
        # TODO: should key conflicts silently overwrite, or warn, or error?
        # TODO: ensure all values are dicts or dict-like?
        for condition_dict in dict_site_conditions:
            if condition_dict is not None:
                site_conditions.update(condition_dict)
        compartment = site_conditions.pop('compartment', self.compartment)
        return MonomerPattern(self, site_conditions, compartment)

    def __repr__(self):
        return  '%s(name=%s, sites=%s, site_states=%s, compartment=%s)' % \
            (self.__class__.__name__, repr(self.name), repr(self.sites), repr(self.site_states), self.compartment and self.compartment.name or None)



class MonomerAny(Monomer):

    clogger.debug('in MonomerAny')

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

    clogger.debug('in MonomerWild')

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

    clogger.debug('in MonomerPattern')

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

    def is_concrete(self):
        """Tests whether all sites in monomer are specified."""
        # assume __init__ did a thorough enough job of error checking that this is is all we need to do
        return len(self.site_conditions) == len(self.monomer.sites)

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

    clogger.debug('in ComplexPattern')

    def __init__(self, monomer_patterns):
        self.monomer_patterns = monomer_patterns

    def is_concrete(self):
        """Tests whether all sites in all of monomer_patterns are specified."""
        return all(mp.is_concrete() for mp in self.monomer_patterns)

    def is_equivalent_to(self, other):
        """Checks for equality with another ComplexPattern"""
        # Didn't implement __eq__ to avoid confusion with __ne__ operator used for Rule building

        # FIXME the literal site_conditions comparison requires bond numbering to be identical,
        #   so some sort of canonicalization of that numbering is necessary.
        if not isinstance(other, ComplexPattern):
            raise Exception("Can only compare ComplexPattern to another ComplexPattern")
        return \
            sorted((mp.monomer, mp.site_conditions) for mp in self.monomer_patterns) == \
            sorted((mp.monomer, mp.site_conditions) for mp in other.monomer_patterns)

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

    clogger.debug('in ReactionPattern')

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



def as_complex_pattern(v):
    """Internal helper to 'upgrade' a MonomerPattern to a ComplexPattern."""

    clogger.debug('in as_complex_pattern')

    if isinstance(v, ComplexPattern):
        return v
    elif isinstance(v, MonomerPattern):
        return ComplexPattern([v])
    else:
        raise InvalidComplexPatternException


def as_reaction_pattern(v):
    """Internal helper to 'upgrade' a Complex- or MonomerPattern to a
    complete ReactionPattern."""

    clogger.debug('in as_reaction_pattern')

    if isinstance(v, ReactionPattern):
        return v
    else:
        try:
            return ReactionPattern([as_complex_pattern(v)])
        except InvalidComplexPatternException:
            raise InvalidReactionPatternException



class Parameter(SelfExporter):

    clogger.debug('in Parameter')

    def __init__(self, name, value=float('nan'), __export=True):
        SelfExporter.__init__(self, name, __export)
        self.value = value

    def __repr__(self):
        return  '%s(name=%s, value=%s)' % (self.__class__.__name__, repr(self.name), repr(self.value))



class Compartment(SelfExporter):

    clogger.debug('in Compartment')

    # FIXME: sane defaults?
    def __init__(self, name, neighbors=[], dimension=3, size=1, __export=True):
        SelfExporter.__init__(self, name, __export)

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

    clogger.debug('in Rule')

    def __init__(self, name, reaction_pattern_set, rate_forward, rate_reverse=None, __export=True):
        SelfExporter.__init__(self, name, __export)

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



class InvalidComplexPatternException(Exception):
    pass

class InvalidReactionPatternException(Exception):
    pass



ANY = MonomerAny()
WILD = MonomerWild()
