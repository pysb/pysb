import sys
import os
import errno
import warnings
import inspect
import re
import collections

def Observe(*args):
    return SelfExporter.default_model.add_observable(*args)

def Initial(*args):
    return SelfExporter.default_model.initial(*args)

def MatchOnce(pattern):
    cp = as_complex_pattern(pattern).copy()
    cp.match_once = True
    return cp


# Internal helper to implement the magic of making model components
# appear in the calling module's namespace.  Do not construct any
# instances; we just use the class for namespace containment.
class SelfExporter(object):

    do_export = True
    default_model = None
    target_globals = None   # the globals dict to which we'll export our symbols
    target_module = None    # the module to which we've exported

    @staticmethod
    def export(obj):
        if not SelfExporter.do_export:
            return
        if not isinstance(obj, (Model, Component)):
            raise Exception("%s is not a type that is understood by SelfExporter" % str(type(obj)))

        # determine the module from which we were called (we need to do this here so we can
        # calculate stacklevel for use in the warning at the bottom of this method)
        cur_module = inspect.getmodule(inspect.currentframe())
        caller_frame = inspect.currentframe()
        # walk up through the stack until we hit a different module
        stacklevel = 1
        while inspect.getmodule(caller_frame) == cur_module:
            stacklevel += 1
            caller_frame = caller_frame.f_back

        # use obj's name as the symbol to export it to (unless modified below)
        export_name = obj.name

        if isinstance(obj, Model):
            if SelfExporter.default_model is not None:
                warnings.warn("Redefining model! (You can probably ignore this if you are running"
                              " code interactively)", ModelExistsWarning, stacklevel);
                # delete previously exported symbols to prevent extra SymbolExistsWarnings
                for name in [c.name for c in SelfExporter.default_model.all_components()] + ['model']:
                    if name in SelfExporter.target_globals:
                        del SelfExporter.target_globals[name]
            SelfExporter.target_module = inspect.getmodule(caller_frame)
            SelfExporter.target_globals = caller_frame.f_globals
            SelfExporter.default_model = obj
            # if not set, assign model's name from the module it lives in. very sneaky and fragile.
            if obj.name is None:
                if SelfExporter.target_module == sys.modules['__main__']:
                    # user ran model .py directly
                    model_filename = inspect.getfile(sys.modules['__main__'])
                    module_name = re.sub(r'\.py$', '', model_filename)
                elif SelfExporter.target_module is not None:
                    # model is imported by some other script (typical case)
                    module_name = SelfExporter.target_module.__name__
                else:
                    # user is defining a model interactively (not really supported, but we'll try)
                    module_name = '<interactive>'
                obj.name = module_name   # internal name for identification
                export_name = 'model'    # symbol name for export
        elif isinstance(obj, Component):
            if SelfExporter.default_model == None:
                raise Exception("A Model must be declared before declaring any model components")
            SelfExporter.default_model.add_component(obj)

        # load obj into target namespace under obj.name
        if SelfExporter.target_globals.has_key(export_name):
            warnings.warn("'%s' already defined" % (export_name), SymbolExistsWarning, stacklevel)
        SelfExporter.target_globals[export_name] = obj


class Component(object):
    """The base class for all the things contained within a model."""

    def __init__(self, name, __export=True):
        self.name = name
        if __export:
            try:
                SelfExporter.export(self)
            except ComponentDuplicateNameError as e:
                # re-raise to hide the stack trace below this point -- it's irrelevant to the user
                # and makes the error harder to understand
                raise e


class Model(object):

    def __init__(self, name=None, __export=True):
        self.name = name
        self.monomers = ComponentSet()
        self.compartments = ComponentSet()
        self.parameters = ComponentSet()
        self.rules = ComponentSet()
        self.species = []
        self.odes = []
        self.observable_patterns = []
        self.observable_groups = {}  # values are tuples of factor,speciesnumber
        self.initial_conditions = []
        if __export:
            SelfExporter.export(self)

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
        try:
            reload(SelfExporter.target_module)
        except SystemError as e:
            # This one specific SystemError occurs when using ipython to 'run' a model .py file
            # directly, then reload()ing the model, which makes no sense anyway. (just re-run it)
            if e.args == ('nameless module',):
                raise Exception('Cannot reload a model which was executed directly in an interactive'
                                'session. Please import the model file as a module instead.')
            else:
                raise
        # return self for "model = model.reload()" idiom, until a better solution can be found
        return SelfExporter.default_model

    def all_components(self):
        components = ComponentSet()
        for container in [self.monomers, self.compartments, self.rules, self.parameters]:
            components |= container
        return components

    def add_component(self, other):
        # We have 4 containers for the 4 types of components. This code determines the right one
        # based on the class of the object being added.  It tries to be defensive against reasonable
        # errors, but still seems sort of fragile.
        container_name = type(other).__name__.lower() + 's'
        container = getattr(self, container_name, None)
        if not isinstance(other, Component) or not isinstance(container, ComponentSet):
            raise Exception("Tried to add component of unknown type '%s' to model" % type(other))
        container.add(other)

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

    def get_species_index(self, complex_pattern):
        # FIXME I don't even want to think about the inefficiency of this, but at least it works
        try:
            return (i for i, s_cp in enumerate(self.species) if s_cp.is_equivalent_to(complex_pattern)).next()
        except StopIteration:
            return None

    def __repr__(self): 
        return "<%s '%s' (monomers: %d, rules: %d, parameters: %d, compartments: %d) at 0x%x>" % \
            (self.__class__.__name__, self.name, len(self.monomers), len(self.rules),
             len(self.parameters), len(self.compartments), id(self))



class Monomer(Component):
    """The Monomer class creates monomers with the specified sites, state-sites, and compartment
    """

    def __init__(self, name, sites=[], site_states={}, __export=True):
        Component.__init__(self, name, __export)

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

    def __call__(self, *args, **kwargs):
        """Build a MonomerPattern object with convenient kwargs for the sites"""
        return MonomerPattern(self, extract_site_conditions(*args, **kwargs), None)

    def __repr__(self):
        return  '%s(name=%s, sites=%s, site_states=%s)' % \
            (self.__class__.__name__, repr(self.name), repr(self.sites), repr(self.site_states))

    

class MonomerAny(Monomer):

    def __init__(self):
        # don't call Monomer.__init__ since this doesn't want
        # Component stuff and has no user-accessible API
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
        # Component stuff and has no user-accessible API
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
        #   a temporary hack - all model components (Component
        #   subclasses?) need weak refs to their parent model.
        return len(self.site_conditions) == len(self.monomer.sites) and \
            (len(SelfExporter.default_model.compartments) == 0 or self.compartment is not None)

    def __call__(self, *args, **kwargs):
        """Build a new MonomerPattern with updated site conditions. Can be used
        to obtain a shallow copy by passing an empty argument list."""
        # The new object will have references to the original monomer and
        # compartment, and a shallow copy of site_conditions which has been
        # updated according to our args (as in Monomer.__call__).
        site_conditions = self.site_conditions.copy()
        site_conditions.update(extract_site_conditions(*args, **kwargs))
        return MonomerPattern(self.monomer, site_conditions, self.compartment)

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
            mp_new = self()
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

    def __init__(self, monomer_patterns, compartment, match_once=False):
        # ensure compartment is a Compartment
        if compartment and not isinstance(compartment, Compartment):
            raise Exception("compartment is not a Compartment object")

        self.monomer_patterns = monomer_patterns
        self.compartment = compartment
        self.match_once = match_once

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

    def copy(self):
        """Implement our own brand of shallow copy.

        The new object will have references to the original compartment, and
        copies of the monomer_patterns."""
        return ComplexPattern([mp() for mp in self.monomer_patterns], self.compartment, self.match_once)

    def __add__(self, other):
        if isinstance(other, ComplexPattern):
            return ReactionPattern([self, other])
        elif isinstance(other, MonomerPattern):
            return ReactionPattern([self, ComplexPattern([other], None)])
        else:
            return NotImplemented

    def __mod__(self, other):
        if isinstance(other, MonomerPattern):
            return ComplexPattern(self.monomer_patterns + [other], None, self.match_once)
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
            cp_new = self.copy()
            cp_new.compartment = other
            return cp_new
        else:
            return NotImplemented

    def __repr__(self):
        ret = ' % '.join([repr(p) for p in self.monomer_patterns])
        if self.compartment is not None:
            ret = '(%s) ** %s' % (ret, self.compartment.name)
        if self.match_once:
            ret = 'MatchOnce(%s)' % ret
        return ret



class ReactionPattern(object):
    """Represents a complete pattern for the product or reactant side
    of a rule.  Essentially a thin wrapper around a list of
    ComplexPatterns."""

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

    if isinstance(v, ComplexPattern):
        return v
    elif isinstance(v, MonomerPattern):
        return ComplexPattern([v], None)
    else:
        raise InvalidComplexPatternException


def as_reaction_pattern(v):
    """Internal helper to 'upgrade' a Complex- or MonomerPattern to a
    complete ReactionPattern."""

    if isinstance(v, ReactionPattern):
        return v
    else:
        try:
            return ReactionPattern([as_complex_pattern(v)])
        except InvalidComplexPatternException:
            raise InvalidReactionPatternException



class Parameter(Component):

    def __init__(self, name, value=float('nan'), __export=True):
        Component.__init__(self, name, __export)
        self.value = value

    def __repr__(self):
        return  '%s(name=%s, value=%s)' % (self.__class__.__name__, repr(self.name), repr(self.value))



class Compartment(Component):
    """The Compartment class expects a "name", "parent", "dimension", and "size" variable from the
    compartment call. name is a string, "parent" should be the name of a defined parent, or None. 
    Dimension should be only 2 (e.g. membranes) or 3 (e.g. cytosol). The size units will depend in the
    manner in which the model variable units have been determined. Note, parent is the compartment object.
    example: Compartment('eCell', dimension=3, size=extraSize, parent=None)
    """

    def __init__(self, name, parent=None, dimension=3, size=None, __export=True):
        Component.__init__(self, name, __export)

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



class Rule(Component):

    def __init__(self, name, reaction_pattern_set, rate_forward, rate_reverse=None,
                 delete_molecules=False,
                 __export=True):
        Component.__init__(self, name, __export)

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
        self.delete_molecules = delete_molecules
        # TODO: ensure all numbered sites are referenced exactly twice within each of reactants and products

    def __repr__(self):
        ret = '%s(name=%s, reactants=%s, products=%s, rate_forward=%s' % \
            (self.__class__.__name__, repr(self.name), repr(self.reactant_pattern), repr(self.product_pattern), repr(self.rate_forward))
        if self.is_reversible:
            ret += ', rate_reverse=%s' % repr(self.rate_reverse)
        if self.delete_molecules:
            ret += ', delete_molecules=True'
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



class ComponentSet(collections.MutableSet, collections.MutableMapping):
    """A container for storing model Components. It behaves mostly like an ordered set, but
    components can also be retrieved and deleted by name by using the [] operator (as in a dict
    lookup)."""
    # The implementation is based on a list instead of a linked list (as OrderedSet is), since the
    # expected usage pattern is heavy on append and retrieve, and light on delete.

    def __init__(self, iterable=[]):
        self._elements = []
        self._map = {}
        for value in iterable:
            self.add(value)

    def __iter__(self):
        return iter(self._elements)

    def __contains__(self, c):
        if not isinstance(c, Component):
            raise TypeError("Can only work with Components, got a %s" % type(c))
        return c.name in self._map and self[c.name] is c

    def __len__(self):
        return len(self._elements)

    def add(self, c):
        if c not in self:
            if c.name in self._map:
                raise ComponentDuplicateNameError("Tried to add a component with a duplicate name: %s" % c.name)
            self._elements.append(c)
            self._map[c.name] = c

    def discard(self, c):
        # TODO
        raise NotImplementedError()

    def __getitem__(self, key):
        return self._map[key]

    def __delitem__(self, key):
        # TODO
        raise NotImplementedError()

    def __setitem__(self, key, value):
        # TODO
        raise NotImplementedError()

    def iterkeys(self):
        for c in self:
            yield c.name

    def itervalues(self):
        return self.__iter__()

    def iteritems(self):
        for c in self:
            yield (c.name, c)

    def keys(self):
        return [c.name for c in self]

    def values(self):
        return [c for c in self]

    def items(self):
        return zip(self.keys(), self)

    def __repr__(self):
        return '{' + \
            ',\n '.join("'%s': %s" % t for t in self.iteritems()) + \
            '}'


class ComponentDuplicateNameError(ValueError):
    """Issued by ComponentSet.add when a component is added with the
    same name as an existing one."""
    pass


def extract_site_conditions(*args, **kwargs):
    """Handle parsing of MonomerPattern site conditions.
    """
    # enforce site conditions as kwargs or a dict but not both
    if (args and kwargs) or len(args) > 1:
        raise Exception("Site conditions may be specified as EITHER keyword arguments OR a single dict")
    # handle normal cases
    elif args:
        site_conditions = args[0].copy()
    else:
        site_conditions = kwargs
    return site_conditions



ANY = MonomerAny()
WILD = MonomerWild()

warnings.simplefilter('always', ModelExistsWarning)
warnings.simplefilter('always', SymbolExistsWarning)
