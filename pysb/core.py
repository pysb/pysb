import sys
import os
import errno
import warnings
import logging
import inspect

logging.basicConfig()
clogger = logging.getLogger("CoreFile")

clogger.info("INITIALIZING")

def Observe(*args):
    return SelfExporter.default_model.add_observable(*args)

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
    target_module = None    # the module to which we've exported

    def __init__(self, name, __export=True):
        self.name = name

        if SelfExporter.do_self_export and __export: #isn't __export always True by the time we get here?

            # determine the module from which we were called
            cur_module = inspect.getmodule(inspect.currentframe())
            caller_frame = inspect.currentframe()
            # walk up through the stack until we hit a different module
            stacklevel = 1
            while inspect.getmodule(caller_frame) == cur_module:
                stacklevel += 1
                caller_frame = caller_frame.f_back

            if isinstance(self, Model):
                if SelfExporter.default_model is not None:
                    warnings.warn("Redefining model! (You can probably ignore this if you are running"
                                  " code interactively)", ModelExistsWarning, stacklevel);
                    # delete previously exported symbols to prevent extra SymbolExistsWarnings
                    for name in [c.name for c in SelfExporter.default_model.all_components()] + ['model']:
                        if name in SelfExporter.target_globals:
                            del SelfExporter.target_globals[name]
                SelfExporter.target_module = inspect.getmodule(caller_frame)
                SelfExporter.target_globals = caller_frame.f_globals
                SelfExporter.default_model = self
                # assign model's name from the module it lives in.  slightly sneaky.
                if self.name is None:
                    self.name = SelfExporter.target_module.__name__
            elif isinstance(self, (Monomer, Compartment, Parameter, Rule)):
                if SelfExporter.default_model == None:
                    raise Exception("A Model must be declared before declaring any model components")
                SelfExporter.default_model.add_component(self)

            # load self into target namespace under self.name
            # FIXME if name already used, add_component will succeed since it's done first.
            #   this whole thing needs to be rethought, really.
            if SelfExporter.target_globals.has_key(name):
                warnings.warn("'%s' already defined" % (name), SymbolExistsWarning, stacklevel)
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

    def reload(self):
        # forcibly removes the .pyc file and reloads the model module
        model_pyc = SelfExporter.target_module.__file__
        if model_pyc[-3:] == '.py':
            model_pyc += 'c'
        try:
            os.unlink(model_pyc)
        except OSError as e:
            # ignore "no such file" errors, re-raise the rest
            if e.errno != errno.ENOENT:
                raise
        reload(SelfExporter.target_module)
        # return self for "model = model.reload()" idiom, until a better solution can be found
        return SelfExporter.default_model

    def all_components(self):
        return self.monomers + self.compartments + self.parameters + self.rules

    def add_component(self, other):
        if isinstance(other, Monomer):
            self.monomers.append(other)
        elif isinstance(other, Compartment):
            self.compartments.append(other)
        elif isinstance(other, Parameter):
            self.parameters.append(other)
            self.parameters.sort(key=lambda p: p.name)  # keep param list sorted
        elif isinstance(other, Rule):
            self.rules.append(other)
        else:
            raise Exception("Tried to add component of unknown type (%s) to model" % type(other))

    # FIXME should this be named add_observable??
    def add_observable(self, name, reaction_pattern):
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

    def get_compartment(self, name):
        # FIXME probably want to store compartments in a dict by name instead of a list
        try:
            return (c for c in self.compartments if c.name == name).next()
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

    def __init__(self, name, sites=[], site_states={}, __export=True):
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

        self.sites = sites
        self.sites_dict = dict.fromkeys(sites)
        self.site_states = site_states

    def __call__(self, *dict_site_conditions, **named_site_conditions):
        """Build a pattern object with convenient kwargs for the sites"""
        site_conditions = named_site_conditions.copy()
        # TODO: should key conflicts silently overwrite, or warn, or error?
        # TODO: ensure all values are dicts or dict-like?
        for condition_dict in dict_site_conditions:
            if condition_dict is not None:
                site_conditions.update(condition_dict)
        return MonomerPattern(self, site_conditions, None)

    def __repr__(self):
        return  '%s(name=%s, sites=%s, site_states=%s)' % \
            (self.__class__.__name__, repr(self.name), repr(self.sites), repr(self.site_states))
    

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
            raise Exception("MonomerPattern with unknown sites in " + str(monomer) + ": " + str(unknown_sites))

        # ensure each value is one of: None, integer, list of integers, string, (string,integer), (string,WILD), ANY
        # FIXME: support state sites
        invalid_sites = []
        for (site, state) in site_conditions.items():
            # pass through to next iteration if state type is ok
            if state == None:
                continue
            elif type(state) == int:
                continue
            elif type(state) == list and all(isinstance(s, int) for s in state):
                continue
            elif type(state) == str:
                continue
            elif type(state) == tuple and type(state[0]) == str and (type(state[1]) == int or state[1] == WILD):
                continue
            elif state == ANY:
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
        """Tests whether all sites and compartment are specified."""
        # assume __init__ did a thorough enough job of error checking that this is is all we need to do
        # FIXME accessing the model via SelfExporter.default_model is
        #   a temporary hack - all model components (SelfExporter
        #   subclasses?) need weak refs to their parent model.
        return len(self.site_conditions) == len(self.monomer.sites) and \
            (len(SelfExporter.default_model.compartments) == 0 or self.compartment is not None)

    def _copy(self):
        """Implement our own brand of semi-deep copy.

        The new object will have references to the original monomer and compartment, and
        a shallow copy of site_conditions."""
        return MonomerPattern(self.monomer, self.site_conditions.copy(), self.compartment)

    def __add__(self, other):
        if isinstance(other, MonomerPattern):
            return ReactionPattern([ComplexPattern([self], None), ComplexPattern([other], None)])
        if isinstance(other, ComplexPattern):
            return ReactionPattern([ComplexPattern([self], None), other])
        else:
            return NotImplemented

    def __mod__(self, other):
        if isinstance(other, MonomerPattern):
            return ComplexPattern([self, other], None)
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

    def __pow__(self, other):
        if isinstance(other, Compartment):
            mp_new = self._copy()
            mp_new.compartment = other
            return mp_new
        else:
            return NotImplemented

    def __repr__(self):
        value = '%s(' % self.monomer.name
        value += ', '.join([
                k + '=' + str(self.site_conditions[k])
                for k in self.monomer.sites
                if self.site_conditions.has_key(k)
                ])
        if self.compartment is not None:
            value += ', compartment=' + self.compartment.name
        value += ')'
        return value



class ComplexPattern(object):
    """Represents a bound set of MonomerPatterns, i.e. a complex.  In
    BNG terms, a list of patterns combined with the '.' operator)."""

    clogger.debug('in ComplexPattern')

    def __init__(self, monomer_patterns, compartment):
        # ensure compartment is a Compartment
        if compartment and not isinstance(compartment, Compartment):
            raise Exception("compartment is not a Compartment object")

        self.monomer_patterns = monomer_patterns
        self.compartment = compartment

    def is_concrete(self):
        """Tests whether all sites in all of monomer_patterns are specified."""
        # FIXME should we also check that self.compartment is None? (BNG rules seem to dictate it)
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

    def _copy(self):
        """Implement our own brand of semi-deep copy.

        The new object will have references to the original compartment, and
        a _copy of the contents of monomer_patterns."""
        return ComplexPattern([mp._copy() for mp in self.monomer_patterns], self.compartment)

    def __add__(self, other):
        if isinstance(other, ComplexPattern):
            return ReactionPattern([self, other])
        elif isinstance(other, MonomerPattern):
            return ReactionPattern([self, ComplexPattern([other], None)])
        else:
            return NotImplemented

    def __mod__(self, other):
        if isinstance(other, MonomerPattern):
            return ComplexPattern(self.monomer_patterns + [other], None)
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

    def __pow__(self, other):
        if isinstance(other, Compartment):
            cp_new = self._copy()
            cp_new.compartment = other
            return cp_new
        else:
            return NotImplemented

    def __repr__(self):
        ret = ' % '.join([repr(p) for p in self.monomer_patterns])
        if self.compartment is not None:
            ret = '(%s) ** %s' % (ret, self.compartment.name)
        return ret



class ReactionPattern(object):
    """Represents a complete pattern for the product or reactant side
    of a rule.  Essentially a thin wrapper around a list of
    ComplexPatterns."""

    clogger.debug('in ReactionPattern')

    def __init__(self, complex_patterns):
        self.complex_patterns = complex_patterns

    def __add__(self, other):
        if isinstance(other, MonomerPattern):
            return ReactionPattern(self.complex_patterns + [ComplexPattern([other], None)])
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
        return ComplexPattern([v], None)
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
    """The Compartment class expects a "name", "parent", "dimension", and "size" variable from the
    compartment call. name is a string, "parent" should be the name of a defined parent, or None. 
    Dimension should be only 2 (e.g. membranes) or 3 (e.g. cytosol). The size units will depend in the
    manner in which the model variable units have been determined. Note, parent is the compartment object.
    example: Compartment('eCell', dimension=3, size=extraSize, parent=None)
    """
    clogger.debug('in Compartment')
   
    def __init__(self, name, parent=None, dimension=3, size=None, __export=True):
        SelfExporter.__init__(self, name, __export)

        if parent != None and isinstance(parent, Compartment) == False:
            raise Exception("parent must be a predefined Compartment or None")
        #FIXME: check for only ONE "None" parent? i.e. only one compartment can have a parent None?

        if size is not None and not isinstance(size, Parameter):
            raise Exception("size must be a parameter (or omitted)")

        self.parent = parent
        self.dimension = dimension
        self.size = size

    def __repr__(self):
        return  '%s(name=%s, parent=%s, dimension=%s, size=%s)' % \
            (self.__class__.__name__, repr(self.name), repr(self.parent), repr(self.dimension), repr(self.size))



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

class ModelExistsWarning(UserWarning):
    """Issued by Model constructor when a second model is defined."""
    pass

class SymbolExistsWarning(UserWarning):
    """Issued by model component constructors when a name is reused."""
    pass



ANY = MonomerAny()
WILD = MonomerWild()

warnings.simplefilter('always', ModelExistsWarning)
warnings.simplefilter('always', SymbolExistsWarning)
