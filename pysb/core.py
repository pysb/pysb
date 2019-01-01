import sys
import os
import errno
import warnings
import inspect
import re
import collections
import weakref
import copy
import itertools
import sympy
import scipy.sparse
import networkx as nx
from collections.abc import Iterable, Mapping, Sequence, Set

from importlib import reload


def MatchOnce(pattern):
    """
    Make a ComplexPattern match-once.

    ``MatchOnce`` adjusts reaction rate multiplicity by only counting a pattern
    match once per species, even if it matches within that species multiple
    times.

    For example, if one were to have molecules of ``A`` degrading with a
    specified rate:

    >>> Rule('A_deg', A() >> None, kdeg)                # doctest: +SKIP

    In the situation where multiple molecules of ``A()`` were present in a
    species (e.g. ``A(a=1) % A(a=1)``), the above ``A_deg`` rule would have
    multiplicity equal to the number of occurences of ``A()`` in the degraded
    species. Thus, ``A(a=1) % A(a=1)`` would degrade twice as fast
    as ``A(a=None)`` under the above rule. If this behavior is not desired,
    the multiplicity can be fixed at one using the ``MatchOnce`` keyword:

    >>> Rule('A_deg', MatchOnce(A()) >> None, kdeg)     # doctest: +SKIP

    """
    cp = as_complex_pattern(pattern).copy()
    cp.match_once = True
    return cp


# A module may define a global with this name (_pysb_doctest_...) to request
# that SelfExporter not issue any ModelExistsWarnings from doctests defined
# therein. (This is the best method we could come up with to manage this
# behavior, as doctest doesn't offer per-doctest setup/teardown.)
_SUPPRESS_MEW = '_pysb_doctest_suppress_modelexistswarning'

class SelfExporter(object):

    """
    Make model components appear in the calling module's namespace.

    This class is for pysb internal use only. Do not construct any instances.

    """
    
    do_export = True
    default_model = None
    target_globals = None   # the globals dict to which we'll export our symbols
    target_module = None    # the module to which we've exported

    @staticmethod
    def export(obj):
        """Export an object by name and add it to the default model."""

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
            new_target_module = inspect.getmodule(caller_frame)
            if SelfExporter.default_model is not None \
                    and new_target_module is SelfExporter.target_module:
                # Warn, unless running a doctest whose containing module set the
                # magic global which tells us to suppress it.
                if not (
                    caller_frame.f_code.co_filename.startswith('<doctest ') and
                    caller_frame.f_globals.get(_SUPPRESS_MEW)):
                    warnings.warn("Redefining model! (You can probably ignore "
                                  "this if you are running code interactively)",
                                  ModelExistsWarning, stacklevel)
                SelfExporter.cleanup()
            SelfExporter.target_module = new_target_module
            SelfExporter.target_globals = caller_frame.f_globals
            SelfExporter.default_model = obj
            # if not set, assign model's name from the module it lives in. very sneaky and fragile.
            if obj.name is None:
                if SelfExporter.target_module == sys.modules['__main__']:
                    # user ran model .py directly
                    model_path = inspect.getfile(sys.modules['__main__'])
                    model_filename = os.path.basename(model_path)
                    module_name = re.sub(r'\.py$', '', model_filename)
                elif SelfExporter.target_module is not None:
                    # model is imported by some other script (typical case)
                    module_name = SelfExporter.target_module.__name__
                else:
                    # user is defining a model interactively (not really supported, but we'll try)
                    module_name = '_interactive_'
                obj.name = module_name   # internal name for identification
                export_name = 'model'    # symbol name for export
        elif isinstance(obj, Component):
            if SelfExporter.default_model is None:
                raise ModelNotDefinedError
            SelfExporter.default_model.add_component(obj)

        # load obj into target namespace under obj.name
        if export_name in SelfExporter.target_globals:
            warnings.warn("'%s' already defined" % (export_name), SymbolExistsWarning, stacklevel)
        SelfExporter.target_globals[export_name] = obj

    @staticmethod
    def add_initial(initial):
        """Add an Initial to the default model."""
        if not SelfExporter.do_export:
            return
        if not isinstance(initial, Initial):
            raise ValueError("initial must be an Initial object")
        if SelfExporter.default_model is None:
            raise ModelNotDefinedError
        SelfExporter.default_model.add_initial(initial)

    @staticmethod
    def cleanup():
        """Delete previously exported symbols."""
        if SelfExporter.default_model is None:
            return
        for name in [c.name for c in SelfExporter.default_model.all_components()] + ['model']:
            if name in SelfExporter.target_globals:
                del SelfExporter.target_globals[name]
        SelfExporter.default_model = None
        SelfExporter.target_globals = None
        SelfExporter.target_module = None

    @staticmethod
    def rename(obj, new_name):
        """Rename a previously exported symbol"""
        if new_name in SelfExporter.target_globals:
            msg = "'%s' already defined" % new_name
            warnings.warn(msg, SymbolExistsWarning, 2)
        if obj.name in SelfExporter.target_globals:
            obj = SelfExporter.target_globals[obj.name]
            SelfExporter.target_globals[new_name] = obj
            del SelfExporter.target_globals[obj.name]
        else:
            raise ValueError("Could not find object in global namespace by its"
                             "name '%s'" % obj.name)


class Symbol(sympy.Dummy):
    def __new__(cls, name, real=True, **kwargs):
        return super(Symbol, cls).__new__(cls, name, real=real, **kwargs)

    def __getnewargs_ex__(self):
        return self.__getnewargs__(), {}

    def _lambdacode(self, printer, **kwargs):
        """ custom printer method that ensures that the dummyid is not
        appended when printing code """
        return self.name


class Component(object):

    """
    The base class for all the named things contained within a model.

    Parameters
    ----------
    name : string
        Name of the component. Must be unique within the containing model.

    Attributes
    ----------
    name : string
        Name of the component.
    model : weakref(Model)
        Containing model.

    """
    _VARIABLE_NAME_REGEX = re.compile(r'[_a-z][_a-z0-9]*\Z', re.IGNORECASE)

    def __init__(self, name, _export=True):
        if not self._VARIABLE_NAME_REGEX.match(name):
            raise InvalidComponentNameError(name)
        self.name = name
        self.model = None  # to be set in Model.add_component
        self._export = _export
        if self._export:
            self._do_export()

        # Try to find calling module by walking the stack
        self._modules = []
        self._function = None
        # We assume we're dealing with Component subclasses here
        frame = inspect.currentframe().f_back
        while frame is not None:
            mod_name = frame.f_globals.get('__name__', '__unnamed__')
            if mod_name in ['IPython.core.interactiveshell', '__main__']:
                break
            if mod_name != 'pysb.core' and not \
                    mod_name.startswith('importlib.'):
                self._modules.append(mod_name)
                if self._function is None:
                    if mod_name == 'pysb.macros':
                        self._function = frame.f_back.f_code.co_name
                    else:
                        self._function = frame.f_code.co_name
            frame = frame.f_back

    def __getstate__(self):
        # clear the weakref to parent model (restored in Model.__setstate__)
        state = self.__dict__.copy()
        state.pop('model', None)
        # Force _export to False; we don't want the unpickling process to
        # trigger SelfExporter.export!
        state['_export'] = False
        return state

    def _do_export(self):
        try:
            SelfExporter.export(self)
        except ComponentDuplicateNameError as e:
            # re-raise to hide the stack trace below this point -- it's irrelevant to the user
            # and makes the error harder to understand
            raise e

    def rename(self, new_name):
        """Change component's name.

        This is typically only needed when deriving one model from another and
        it would be desirable to change a component's name in the derived
        model."""
        if self.model:
            self.model()._rename_component(self, new_name)
        if self._export:
            SelfExporter.rename(self, new_name)
        self.name = new_name


class Monomer(Component):
    """
    Model component representing a protein or other molecule.

    Parameters
    ----------
    sites : list of strings, optional
        Names of the sites.
    site_states : dict of string => string, optional
        Allowable states for sites. Keys are sites and values are lists of
        states. Sites which only take part in bond formation and never take on a
        state may be omitted.

    Attributes
    ----------
    Identical to Parameters (see above).

    Notes
    -----

    A Monomer instance may be \"called\" like a function to produce a
    MonomerPattern, as syntactic sugar to approximate rule-based modeling
    language syntax. It is typically called with keyword arguments where the arg
    names are sites and values are site conditions such as bond numbers or
    states (see the Notes section of the :py:class:`MonomerPattern`
    documentation for details). To help in situations where kwargs are unwieldy
    (for example if a site name is computed dynamically or stored in a variable)
    a dict following the same layout as the kwargs may be passed as the first
    and only positional argument instead.

    Site names and state values must start with a letter, or one or more
    underscores followed by a letter. Any remaining characters must be
    alphanumeric or underscores.
    """
    def __init__(self, name, sites=None, site_states=None, _export=True):
        # Create default empty containers.
        if sites is None:
            sites = []
        if site_states is None:
            site_states = {}

        # ensure sites is some kind of list (presumably of strings) but not a
        # string itself
        if not isinstance(sites, Iterable) or \
               isinstance(sites, str):
            raise ValueError("sites must be a list of strings")

        # ensure no duplicate sites and validate each site name
        sites_seen = {}
        for site in sites:
            if not self._VARIABLE_NAME_REGEX.match(site):
                raise ValueError('Invalid site name: ' + str(site))
            sites_seen.setdefault(site, 0)
            sites_seen[site] += 1

        # ensure site_states keys are all known sites
        unknown_sites = [site for site in site_states if not site in sites_seen]
        if unknown_sites:
            raise ValueError("Unknown sites in site_states: " +
                             str(unknown_sites))
        # ensure site_states values are all strings
        invalid_sites = [site for (site, states) in site_states.items()
                         if not all([isinstance(s, str)
                                     and self._VARIABLE_NAME_REGEX.match(s)
                                     for s in states])]
        if invalid_sites:
            raise ValueError("Invalid or non-string state values in "
                             "site_states for sites: " + str(invalid_sites))

        self.sites = list(sites)
        self.site_states = site_states
        Component.__init__(self, name, _export)

    def __call__(self, conditions=None, **kwargs):
        """
        Return a MonomerPattern object based on this Monomer.

        See the Notes section of this class's documentation for details.

        Parameters
        ----------
        conditions: dict, optional
            See MonomerPattern.site_conditions.
        **kwargs: Union[None, int, str, Tuple[str,int], MultiSite, List[int]]
            See MonomerPattern.site_conditions.

        """
        return MonomerPattern(self, extract_site_conditions(conditions,
                                                            **kwargs), None)

    def __repr__(self):
        value = '%s(%s' % (self.__class__.__name__, repr(self.name))
        if self.sites:
            value += ', %s' % repr(self.sites)
        if self.site_states:
            value += ', %s' % repr(self.site_states)
        value += ')'
        return value


def _check_state(monomer, site, state):
    """ Check a monomer site allows the specified state """
    if state not in monomer.site_states[site]:
        args = state, monomer.name, site, monomer.site_states[site]
        template = "Invalid state choice '{}' in Monomer {}, site {}. Valid " \
                   "state choices: {}"
        raise ValueError(template.format(*args))
    return True


def _check_bond(bond):
    """ A bond can either by a single int, WILD, ANY, or a list of ints """
    return (
        isinstance(bond, int)
        or bond is WILD
        or bond is ANY
        or isinstance(bond, list) and all(isinstance(b, int) for b in bond)
    )


def is_state_bond_tuple(state):
    """ Check the argument is a (state, bond) tuple for a Mononer site """
    return (
        isinstance(state, tuple)
        and len(state) == 2
        and isinstance(state[0], str)
        and _check_bond(state[1])
    )


def _check_state_bond_tuple(monomer, site, state):
    """ Check that 'state' is a (state, bond) tuple, and validate the state """
    return is_state_bond_tuple(state) and _check_state(monomer, site, state[0])


def validate_site_value(state, monomer=None, site=None, _in_multistate=False):
    if state is None:
        return True
    elif isinstance(state, str):
        if monomer and site:
            if not _check_state(monomer, site, state):
                return False
        return True
    elif _check_bond(state):
        return True
    elif is_state_bond_tuple(state):
        if monomer and site:
            _check_state(monomer, site, state[0])
        return True
    elif isinstance(state, MultiState):
        if _in_multistate:
            raise ValueError('Cannot nest MultiState within each other')

        if monomer and site:
            site_counts = collections.Counter(monomer.sites)
            if len(state) > site_counts[site]:
                raise ValueError(
                    'MultiState for site "{}" on monomer "{}" has maximum '
                    'length {}'.format(site, monomer.name, site_counts[site])
                )

            return all(validate_site_value(s, monomer, site, True) for s in
                       state)

        return True
    else:
        return False


class MultiState(object):
    """
    MultiState for a Monomer (also known as duplicate sites)

    MultiStates are duplicate copies of a site which each have the same name and
    semantics. In BioNetGen, these are known as duplicate sites. MultiStates
    are not supported by Kappa.

    When declared, a MultiState instance is not connected to any Monomer or
    site, so full validation is deferred until it is used as part of a
    :py:class:`MonomerPattern` or :py:class:`ComplexPattern`.

    Examples
    --------

    Define a Monomer "A" with MultiState "a", which has two copies, and
    Monomer "B" with MultiState "b", which also has two copies but can take
    state values "u" and "p":

    >>> Model()  # doctest:+ELLIPSIS
    <Model '_interactive_' (monomers: 0, ...
    >>> Monomer('A', ['a', 'a'])  # BNG: A(a, a)
    Monomer('A', ['a', 'a'])
    >>> Monomer('B', ['b', 'b'], {'b': ['u', 'p']})  # BNG: B(b~u~p, b~u~p)
    Monomer('B', ['b', 'b'], {'b': ['u', 'p']})

    To specify MultiStates, use the MultiState class. Here are some valid
    examples of MultiState patterns, with their BioNetGen equivalents:

    >>> A(a=MultiState(1, 2))  # BNG: A(a!1,a!2)
    A(a=MultiState(1, 2))
    >>> B(b=MultiState('u', 'p'))  # BNG: A(A~u,A~p)
    B(b=MultiState('u', 'p'))
    >>> A(a=MultiState(1, 2)) % B(b=MultiState(('u', 1), 2))  # BNG: A(a!1, a!2).B(b~u!1, b~2)
    A(a=MultiState(1, 2)) % B(b=MultiState(('u', 1), 2))
    """
    def __init__(self, *args):
        if len(args) == 1:
            raise ValueError('MultiState should not be used when only a single '
                             'site is specified')
        self.sites = args
        for s in self.sites:
            validate_site_value(s, _in_multistate=True)

    def __len__(self):
        return len(self.sites)

    def __iter__(self):
        return iter(self.sites)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join(
            repr(s) for s in self))


class MonomerPattern(object):

    """
    A pattern which matches instances of a given monomer.

    Parameters
    ----------
    monomer : Monomer
        The monomer to match.
    site_conditions : dict
        The desired state of the monomer's sites. Keys are site names and values
        are described below in Notes.
    compartment : Compartment or None
        The desired compartment where the monomer should exist. None means
        \"don't-care\".

    Attributes
    ----------
    Identical to Parameters (see above).

    Notes
    -----
    The acceptable values in the `site_conditions` dict are as follows:

    * ``None`` : no bond
    * *str* : state
    * *int* : a bond (to a site with the same number in a ComplexPattern)
    * *list of int* : multi-bond (not valid in Kappa)
    * ``ANY`` : \"any\" bond (bound to something, but don't care what)
    * ``WILD`` : \"wildcard\" bond (bound or not bound)
    * *tuple of (str, int)* : state with specified bond
    * *tuple of (str, WILD)* : state with wildcard bond
    * *tuple of (str, ANY)* : state with any bond
    * MultiState : duplicate sites

    If a site is not listed in site_conditions then the pattern will match any
    state for that site, i.e. \"don't write, don't care\".

    """
    def __init__(self, monomer, site_conditions, compartment):
        # ensure all keys in site_conditions are sites in monomer
        unknown_sites = [site for site in site_conditions
                              if site not in monomer.sites]
        if unknown_sites:
            raise Exception("MonomerPattern with unknown sites in " +
                            str(monomer) + ": " + str(unknown_sites))

        invalid_sites = []
        for (site, state) in site_conditions.items():
            if not validate_site_value(state, monomer, site):
                invalid_sites.append(site)
        if invalid_sites:
            raise ValueError("Invalid state value for sites: " +
                             '; '.join(['%s=%s' % (s, str(site_conditions[s]))
                                       for s in invalid_sites]) +
                             ' in {}'.format(monomer))

        # ensure compartment is a Compartment
        if compartment and not isinstance(compartment, Compartment):
            raise ValueError("compartment is not a Compartment object")

        self.monomer = monomer
        self.site_conditions = site_conditions
        self.compartment = compartment
        self._graph = None
        self._tag = None

    def is_concrete(self):
        """
        Return a bool indicating whether the pattern is 'concrete'.

        'Concrete' means the pattern satisfies ALL of the following:

        1. All sites have specified conditions
        2. If the model uses compartments, the compartment is specified.

        """
        # 1.
        sites_ok = self.is_site_concrete()
        # 2.
        compartment_ok = not self.monomer.model().compartments or self.compartment
        return compartment_ok and sites_ok

    def is_site_concrete(self):
        """
        Return a bool indicating whether the pattern is 'site-concrete'.

        'Site-concrete' means all sites have specified conditions."""
        dup_sites = {k: v for k, v in
                     collections.Counter(self.monomer.sites).items() if v > 1}
        if len(self.site_conditions) != len(self.monomer.sites) and \
                not dup_sites:
            return False
        for site_name, site_val in self.site_conditions.items():
            if site_name in dup_sites:
                if not isinstance(site_val, MultiState) or \
                        len(site_val) < dup_sites[site_name]:
                    return False

                if not all(self._site_instance_concrete(site_name, s)
                           for s in site_val):
                    return False
            elif not self._site_instance_concrete(site_name, site_val):
                return False

        return True

    def _site_instance_concrete(self, site_name, site_val):
        if isinstance(site_val, str):
            site_state = site_val
            site_bond = None
        elif isinstance(site_val, tuple):
            site_state, site_bond = site_val
        else:
            site_bond = site_val
            site_state = None

        if site_bond is ANY or site_bond is WILD:
            return False
        if site_state is None and site_name in \
                self.monomer.site_states.keys():
            return False

        return True

    def _as_graph(self):
        """
        Convert MonomerPattern to networkx graph, caching the result

        See :func:`ComplexPattern._as_graph` for implementation details
        """
        if self._graph is None:
            self._graph = as_complex_pattern(self)._as_graph()

        return self._graph

    def __call__(self, conditions=None, **kwargs):
        """Build a new MonomerPattern with updated site conditions. Can be used
        to obtain a shallow copy by passing an empty argument list."""
        # The new object will have references to the original monomer and
        # compartment, and a shallow copy of site_conditions which has been
        # updated according to our args (as in Monomer.__call__).
        site_conditions = self.site_conditions.copy()
        site_conditions.update(extract_site_conditions(conditions, **kwargs))
        mp = MonomerPattern(self.monomer, site_conditions, self.compartment)
        mp._tag = self._tag
        return mp

    def __add__(self, other):
        if isinstance(other, MonomerPattern):
            return ReactionPattern([ComplexPattern([self], None), ComplexPattern([other], None)])
        if isinstance(other, ComplexPattern):
            return ReactionPattern([ComplexPattern([self], None), other])
        elif other is None:
            rp = as_reaction_pattern(self)
            rp.complex_patterns.append(None)
            return rp
        else:
            return NotImplemented

    def __radd__(self, other):
        if other is None:
            rp = as_reaction_pattern(self)
            rp.complex_patterns = [None] + rp.complex_patterns
            return rp
        else:
            return NotImplemented

    def __mod__(self, other):
        if isinstance(other, MonomerPattern):
            return ComplexPattern([self, other], None)
        else:
            return NotImplemented

    def __rshift__(self, other):
        return build_rule_expression(self, other, False)

    def __rrshift__(self, other):
        return build_rule_expression(other, self, False)

    def __or__(self, other):
        return build_rule_expression(self, other, True)

    def __ror__(self, other):
        return build_rule_expression(other, self, True)

    def __ne__(self, other):
        warnings.warn("'<>' for reversible rules will be removed in a future "
                      "version of PySB. Use '|' instead.",
                      DeprecationWarning,
                      stacklevel=2)
        return self.__or__(other)

    def __pow__(self, other):
        if isinstance(other, Compartment):
            if self.compartment is not None:
                raise CompartmentAlreadySpecifiedError()
            mp_new = self()
            mp_new.compartment = other
            return mp_new
        else:
            return NotImplemented

    def __matmul__(self, other):
        if not isinstance(other, Tag):
            return NotImplemented

        if self._tag:
            raise TagAlreadySpecifiedError()

        # Need to upgrade to a ComplexPattern
        cp_new = as_complex_pattern(self)
        cp_new._tag = other
        return cp_new

    def __repr__(self):
        value = '%s(' % self.monomer.name
        sites_unique = list(collections.OrderedDict.fromkeys(
            self.monomer.sites))
        value += ', '.join([
                k + '=' + repr(self.site_conditions[k])
                for k in sites_unique
                if k in self.site_conditions
                ])
        value += ')'
        if self.compartment is not None:
            value += ' ** ' + self.compartment.name
        if self._tag:
            value = '{} @ {}'.format(self._tag.name, value)
        return value


class ComplexPattern(object):

    """
    A bound set of MonomerPatterns, i.e. a pattern to match a complex.

    In BNG terms, a list of patterns combined with the '.' operator.

    Parameters
    ----------
    monomer_patterns : list of MonomerPatterns
        MonomerPatterns that make up the complex.
    compartment : Compartment or None
        Location restriction. None means don't care.
    match_once : bool, optional
        If True, the pattern will only count once against a species in which the
        pattern can match the monomer graph in multiple distinct ways. If False
        (default), the pattern will count as many times as it matches the
        monomer graph, leading to a faster effective reaction rate.

    Attributes
    ----------
    Identical to Parameters (see above).

    """

    def __init__(self, monomer_patterns, compartment, match_once=False):
        # ensure compartment is a Compartment
        if compartment and not isinstance(compartment, Compartment):
            raise Exception("compartment is not a Compartment object")

        # Drop species cpt, if redundant
        if compartment and len(monomer_patterns) == 1 and \
                monomer_patterns[0].compartment == compartment:
            compartment = None

        self.monomer_patterns = monomer_patterns
        self.compartment = compartment
        self.match_once = match_once
        self._graph = None
        self._tag = None

    def is_concrete(self):
        """
        Return a bool indicating whether the pattern is 'concrete'.

        'Concrete' means the pattern satisfies ANY of the following:
        1. All monomer patterns are concrete
        2. The compartment is specified AND all monomer patterns are site-concrete
        """
        # 1.
        mp_concrete_ok = all(mp.is_concrete() for mp in self.monomer_patterns)
        # 2.
        compartment_ok = self.compartment is not None and \
            all(mp.is_site_concrete() for mp in self.monomer_patterns)
        return mp_concrete_ok or compartment_ok

    def _as_graph(self):
        """
        Return the ComplexPattern represented as a networkx graph

        ComplexPatterns can be represented as a graph. This is mainly useful
        for comparing if ComplexPatterns are equivalent (see
        :func:`ComplexPattern.is_equivalent_to`).

        It turns out this is non-trivial because 1) bond numbering is
        arbitrary and 2) ComplexPatterns can contain MonomerPatterns which
        are identical. The latter problem makes it impossible to merely
        order the MonomerPatterns using a canonical ordering for comparison,
        while ensuring correctness in all cases [Blinov2006]_.

        We solve the problem using broadly the same approach as BioNetGen -
        encode each complex pattern as a graph and check if they are
        isomorphic to each other [Faeder2009]_. However, our approach
        differs in that we do not need to use a hierarchical graph like
        BioNetGen's hnauty algorithm. We use networkx, in which graph nodes are
        Python objects rather than strings; thus, we ensure that
        monomers/sites/states with the same name are not evaluated to be
        equal, because they have different object type.

        **Implementation details**
        Each monomer, site, state and compartment is represented as a node.
        Edges represent bonds (when between sites), or a relationship
        (monomers have sites, sites have states, MonomerPatterns and
        ComplexPatterns can have Compartments). A special "no bond" node is
        used to denote that the connected site is unbound; this is necessary
        because pattern matching is performed by checking for an isomorphic
        subgraph, and we need to distinguish between explicitly unbound and
        unspecified bond (equivalent to the `ANY` keyword).

        Internally, networkx references nodes using an integer. We use a
        private autoincrementing integer generator function `autoinc` to track
        nodes, but this is not used when checking graph isomorphism (instead,
        node to node object equality is checked).

        The `WILD` keyword should match any bond except the special "no
        bond" node - as special private `WildTester` function is used for
        this purpose.

        Compartment nodes are tracked and kept unique by the private
        `add_or_get_compartment_node` function, which uses a dictionary to
        track Compartment->node_id mapping.

        .. [Blinov2006] https://link.springer.com/chapter/10.1007%2F11905455_5
        .. [Faeder2009] https://www.csb.pitt.edu/Faculty/Faeder/Publications/Reprints/Faeder_2009.pdf
        """
        if self._graph is not None:
            return self._graph

        NO_BOND = 'NoBond'

        def autoinc():
            i = 0
            while True:
                yield i
                i += 1
        node_count = autoinc()

        class AnyBondTester(object):
            def __eq__(self, other):
                return not isinstance(other, Component) and other != NO_BOND

        any_bond_tester = AnyBondTester()

        bond_edges = collections.defaultdict(list)
        g = nx.Graph()
        _cpt_nodes = {}

        def add_or_get_compartment_node(cpt):
            try:
                return _cpt_nodes[cpt]
            except KeyError:
                cpt_node_id = next(node_count)
                _cpt_nodes[cpt] = cpt_node_id
                g.add_node(cpt_node_id, id=cpt)
                return cpt_node_id

        species_cpt_node_id = None
        if self.compartment:
            species_cpt_node_id = add_or_get_compartment_node(self.compartment)

        def _handle_site_instance(state_or_bond):
            mon_site_id = next(node_count)
            g.add_node(mon_site_id, id=site)
            g.add_edge(mon_node_id, mon_site_id)
            state = None
            bond_num = None
            if state_or_bond is WILD:
                return
            elif isinstance(state_or_bond, str):
                state = state_or_bond
            elif is_state_bond_tuple(state_or_bond):
                state = state_or_bond[0]
                bond_num = state_or_bond[1]
            elif isinstance(state_or_bond, (int, list)):
                bond_num = state_or_bond
            elif state_or_bond is not ANY and state_or_bond is not None:
                raise ValueError('Unrecognized state: {}'.format(
                    state_or_bond))

            if state_or_bond is ANY or bond_num is ANY:
                bond_num = any_bond_tester
                any_bond_tester_id = next(node_count)
                g.add_node(any_bond_tester_id, id=any_bond_tester)
                g.add_edge(mon_site_id, any_bond_tester_id)

            if state is not None:
                mon_site_state_id = next(node_count)
                g.add_node(mon_site_state_id, id=state)
                g.add_edge(mon_site_id, mon_site_state_id)

            if bond_num is None:
                bond_edges[NO_BOND].append(mon_site_id)
            elif isinstance(bond_num, int):
                bond_edges[bond_num].append(mon_site_id)
            elif isinstance(bond_num, list):
                for bond in bond_num:
                    bond_edges[bond].append(mon_site_id)

        for mp in self.monomer_patterns:
            mon_node_id = next(node_count)
            g.add_node(mon_node_id, id=mp.monomer)
            if mp.compartment or self.compartment:
                cpt_node_id = add_or_get_compartment_node(mp.compartment or
                                                          self.compartment)
                g.add_edge(mon_node_id, cpt_node_id)

            for site, state_or_bond in mp.site_conditions.items():
                if isinstance(state_or_bond, MultiState):
                    # Duplicate sites
                    [_handle_site_instance(s) for s in state_or_bond]
                else:
                    _handle_site_instance(state_or_bond)

        # Unbound edges
        unbound_sites = bond_edges.pop(NO_BOND, None)
        if unbound_sites is not None:
            no_bond_id = next(node_count)
            g.add_node(no_bond_id, id=NO_BOND)
            for unbound_site in unbound_sites:
                g.add_edge(unbound_site, no_bond_id)

        # Add bond edges
        for site_nodes in bond_edges.values():
            if len(site_nodes) == 1:
                # Treat dangling bond as WILD
                any_bond_tester_id = next(node_count)
                g.add_node(any_bond_tester_id, id=any_bond_tester)
                g.add_edge(site_nodes[0], any_bond_tester_id)
            for n1, n2 in itertools.combinations(site_nodes, 2):
                g.add_edge(n1, n2)

        # Remove the species compartment if all monomer nodes have a
        # compartment
        if species_cpt_node_id is not None and \
                        g.degree(species_cpt_node_id) == 0:
            g.remove_node(species_cpt_node_id)

        self._graph = g
        return self._graph

    def is_equivalent_to(self, other):
        """
        Test a concrete ComplexPattern for equality with another.

        Use of this method on non-concrete ComplexPatterns was previously
        allowed, but is now deprecated.
        """
        from pysb.pattern import match_complex_pattern
        # Didn't implement __eq__ to avoid confusion with __ne__ operator used
        # for Rule building

        # Check both patterns are concrete
        if not self.is_concrete() or not other.is_concrete():
            warnings.warn("is_equivalent_to() will only work with concrete "
                          "patterns in a future version", DeprecationWarning)

        return match_complex_pattern(self, other, exact=True)

    def matches(self, other):
        """
        Compare another ComplexPattern against this one

        Parameters
        ----------
        other: ComplexPattern
            A ComplexPattern to match against self

        Returns
        -------
        bool
            True if other matches self; False otherwise.

        """
        if not self.is_concrete():
            raise ValueError('matches() requires self to be a concrete '
                             'pattern')
        from pysb.pattern import match_complex_pattern
        return match_complex_pattern(other, self, exact=False)

    def copy(self):
        """
        Implement our own brand of shallow copy.

        The new object will have references to the original compartment, and
        copies of the monomer_patterns.
        """
        cp = ComplexPattern([mp() for mp in self.monomer_patterns],
                            self.compartment,
                            self.match_once)
        cp._tag = self._tag
        return cp

    def __call__(self, conditions=None, **kwargs):
        """Build a new ComplexPattern with updated site conditions."""

        kwargs = extract_site_conditions(conditions, **kwargs)

        # Ensure we don't have more than one of any Monomer in our patterns.
        mon_counts = collections.Counter(mp.monomer.name for mp in
                                         self.monomer_patterns)
        dup_monomers = [mon for mon, count in mon_counts.items() if count > 1]
        if dup_monomers:
            raise DuplicateMonomerError("ComplexPattern has duplicate "
                                        "Monomers: " + str(dup_monomers))

        # Ensure all specified sites are present in some Monomer.
        self_site_groups = (mp.monomer.sites for mp in self.monomer_patterns)
        self_sites = list(itertools.chain(*self_site_groups))
        unknown_sites = set(kwargs).difference(self_sites)
        if unknown_sites:
            raise UnknownSiteError("Unknown sites in argument list: " +
                                   ", ".join(unknown_sites))

        # Ensure no specified site is present in multiple Monomers.
        used_sites = [s for s in self_sites if s in kwargs]
        sgroups = itertools.groupby(sorted(used_sites))
        scounts = [(name, sum(1 for s in sites)) for name, sites in sgroups]
        dup_sites = [name for name, count in scounts if count > 1]
        if dup_sites:
            raise DuplicateSiteError("ComplexPattern has duplicate sites: " +
                                     str(dup_sites))

        # Copy self so we can modify it in place before returning it.
        cp = self.copy()
        # Build map from site name to MonomerPattern.
        site_map = {}
        for mp in cp.monomer_patterns:
            site_map.update(dict.fromkeys(mp.monomer.sites, mp))
        # Apply kwargs to our ComplexPatterns.
        for site, condition in kwargs.items():
            site_map[site].site_conditions[site] = condition
        return cp

    def __add__(self, other):
        if isinstance(other, ComplexPattern):
            return ReactionPattern([self, other])
        elif isinstance(other, MonomerPattern):
            return ReactionPattern([self, ComplexPattern([other], None)])
        elif other is None:
            rp = as_reaction_pattern(self)
            rp.complex_patterns.append(None)
            return rp
        else:
            return NotImplemented

    def __radd__(self, other):
        if other is None:
            rp = as_reaction_pattern(self)
            rp.complex_patterns = [None] + rp.complex_patterns
            return rp
        else:
            return NotImplemented

    def __mod__(self, other):
        if self._tag:
            raise ValueError('Tag should be specified at the end of the complex')
        if isinstance(other, MonomerPattern):
            return ComplexPattern(self.monomer_patterns + [other], self.compartment, self.match_once)
        elif isinstance(other, ComplexPattern):
            if self.compartment is not other.compartment:
                raise ValueError("merged ComplexPatterns must specify the same compartment")
            elif self.match_once != other.match_once:
                raise ValueError("merged ComplexPatterns must have the same value of match_once")
            return ComplexPattern(self.monomer_patterns + other.monomer_patterns, self.compartment, self.match_once)
        else:
            return NotImplemented

    def __rmod__(self, other):
        if isinstance(other, MonomerPattern):
            return ComplexPattern([other] + self.monomer_patterns, self.compartment, self.match_once)
        else:
            return NotImplemented

    def __rshift__(self, other):
        return build_rule_expression(self, other, False)

    def __rrshift__(self, other):
        return build_rule_expression(other, self, False)

    def __or__(self, other):
        return build_rule_expression(self, other, True)

    def __ror__(self, other):
        return build_rule_expression(other, self, True)

    def __ne__(self, other):
        warnings.warn("'<>' for reversible rules will be removed in a future "
                      "version of PySB. Use '|' instead.",
                      DeprecationWarning,
                      stacklevel=2)
        return self.__or__(other)

    def __pow__(self, other):
        if isinstance(other, Compartment):
            if self.compartment is not None:
                raise CompartmentAlreadySpecifiedError()
            cp_new = self.copy()
            cp_new.compartment = other
            return cp_new
        else:
            return NotImplemented

    def __matmul__(self, other):
        if not isinstance(other, Tag):
            return NotImplemented

        if self._tag:
            raise TagAlreadySpecifiedError()

        cp_new = self.copy()
        cp_new._tag = other
        return cp_new

    def __repr__(self):
        # Monomer patterns need to be in parentheses if they have a tag,
        # except in the first position, to preserve operator precedence
        ret = ' % '.join(
            [repr(p)
             if idx == 0 or p._tag is None
             else '({})'.format(repr(p))
             for idx, p in enumerate(self.monomer_patterns)])
        if self.compartment:
            if len(self.monomer_patterns) > 1:
                ret = '(%s)' % ret
            ret += ' ** %s' % self.compartment.name
        if self.match_once:
            ret = 'MatchOnce(%s)' % ret
        if self._tag:
            ret = '{} @ {}'.format(ret, self._tag.name)
        return ret


class ReactionPattern(object):

    """
    A pattern for the entire product or reactant side of a rule.

    Essentially a thin wrapper around a list of ComplexPatterns. In BNG terms, a
    list of complex patterns combined with the '+' operator.

    Parameters
    ----------
    complex_patterns : list of ComplexPatterns
        ComplexPatterns that make up the reaction pattern.

    Attributes
    ----------
    Identical to Parameters (see above).

    """

    def __init__(self, complex_patterns):
        self.complex_patterns = complex_patterns
        from pysb.pattern import check_dangling_bonds
        check_dangling_bonds(self)

    def __add__(self, other):
        if isinstance(other, MonomerPattern):
            return ReactionPattern(self.complex_patterns + [ComplexPattern([other], None)])
        elif isinstance(other, ComplexPattern):
            return ReactionPattern(self.complex_patterns + [other])
        elif other is None:
            self.complex_patterns.append(None)
            return self
        else:
            return NotImplemented

    def __radd__(self, other):
        if other is None:
            self.complex_patterns = [None] + self.complex_patterns
            return self
        else:
            return NotImplemented

    def __rshift__(self, other):
        """Irreversible reaction"""
        return build_rule_expression(self, other, False)

    def __rrshift__(self, other):
        return build_rule_expression(other, self, False)

    def __or__(self, other):
        return build_rule_expression(self, other, True)

    def __ne__(self, other):
        warnings.warn("'<>' for reversible rules will be removed in a future "
                      "version of PySB. Use '|' instead.",
                      DeprecationWarning,
                      stacklevel=2)
        return self.__or__(other)

    def __repr__(self):
        if len(self.complex_patterns):
            return ' + '.join([repr(p) for p in self.complex_patterns])
        else:
            return 'None'

    def matches(self, other):
        """
        Match the 'other' ReactionPattern against this one

        See :func:`pysb.pattern.match_reaction_pattern` for details
        """
        from pysb.pattern import match_reaction_pattern
        return match_reaction_pattern(other, self)


class RuleExpression(object):

    """
    A container for the reactant and product patterns of a rule expression.

    Contains one ReactionPattern for each of reactants and products, and a bool
    indicating reversibility. This is a temporary object used to implement
    syntactic sugar through operator overloading. The Rule constructor takes an
    instance of this class as its first argument, but simply extracts its fields
    and discards the object itself.

    Parameters
    ----------
    reactant_pattern, product_pattern : ReactionPattern
        The reactants and products of the rule.
    is_reversible : bool
        If True, the reaction is reversible. If False, it's irreversible.

    Attributes
    ----------
    Identical to Parameters (see above).

    """

    def __init__(self, reactant_pattern, product_pattern, is_reversible):
        self.reactant_pattern = reactant_pattern
        self.product_pattern = product_pattern
        self.is_reversible = is_reversible

    def __repr__(self):
        operator = '|' if self.is_reversible else '>>'
        return '%s %s %s' % (repr(self.reactant_pattern), operator,
                             repr(self.product_pattern))


def as_complex_pattern(v):
    """Internal helper to 'upgrade' a MonomerPattern to a ComplexPattern."""
    if isinstance(v, ComplexPattern):
        return v
    elif isinstance(v, Monomer):
        return ComplexPattern([v()], None)
    elif isinstance(v, MonomerPattern):
        return ComplexPattern([v], None)
    else:
        raise InvalidComplexPatternException


def as_reaction_pattern(v):
    """Internal helper to 'upgrade' a Complex- or MonomerPattern or None to a
    complete ReactionPattern."""
    if isinstance(v, ReactionPattern):
        return v
    elif v is None:
        return ReactionPattern([])
    else:
        try:
            return ReactionPattern([as_complex_pattern(v)])
        except InvalidComplexPatternException:
            raise InvalidReactionPatternException


def build_rule_expression(reactant, product, is_reversible):
    """Internal helper for operators which return a RuleExpression."""
    # Make sure the types of both reactant and product are acceptable.
    try:
        reactant = as_reaction_pattern(reactant)
        product = as_reaction_pattern(product)
    except InvalidReactionPatternException:
        return NotImplemented
    # Synthesis/degradation rules cannot be reversible.
    if (reactant is None or product is None) and is_reversible:
        raise InvalidReversibleSynthesisDegradationRule
    return RuleExpression(reactant, product, is_reversible)


class Parameter(Component, Symbol):

    """
    Model component representing a named constant floating point number.

    Parameters are used as reaction rate constants, compartment volumes and
    initial (boundary) conditions for species.

    Parameters
    ----------
    value : number, optional
        The numerical value of the parameter. Defaults to 0.0 if not specified.
        The provided value is converted to a float before being stored, so any
        value that cannot be coerced to a float will trigger an exception.
    nonnegative : bool, optional
        Sets the assumption whether this parameter is nonnegative (>=0).
        Affects simplifications of expressions that involve this parameter.
        By default, parameters are assumed to be non-negative.
    integer : bool, optional
        Sets the assumption whether this parameter takes integer values,
        which affects simplifications of expressions that involve this
        parameter. By default, parameters are not assumed to take integer values.

    Attributes
    ----------
    value (see Parameters above).

    """

    def __new__(cls, name, value=0.0, _export=True, nonnegative=True,
                integer=False):
        return super(Parameter, cls).__new__(cls, name, real=True,
                                             nonnegative=nonnegative,
                                             integer=integer)

    def __getnewargs__(self):
        return (self.name, self.value, False, self.assumptions0['nonnegative'],
                self.assumptions0['integer'])

    def __init__(self, name, value=0.0, _export=True, nonnegative=True,
                 integer=False):
        self.value = value
        Component.__init__(self, name, _export)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self.check_value(new_value)
        self._value = float(new_value)
    
    def get_value(self):
        return self.value

    def check_value(self, value):
        if self.is_integer:
            if not float(value).is_integer():
                raise ValueError('Cannot assign an non-integer value to a '
                                 'parameter assumed to be an integer')
        if self.is_nonnegative:
            if float(value) < 0:
                raise ValueError('Cannot assign a negative value to a '
                                 'parameter assumed to be nonnegative')

    def __repr__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, repr(self.name), repr(self.value))

    def __str__(self):
        return repr(self)


class Compartment(Component):

    """
    Model component representing a bounded reaction volume.

    Parameters
    ----------
    parent : Compartment, optional
        Compartment which contains this one. If not specified, this will be the
        outermost compartment and its parent will be set to None.
    dimension : integer, optional
        The number of spatial dimensions in the compartment, either 2 (i.e. a
        membrane) or 3 (a volume).
    size : Parameter or Expression, optional
        A parameter or constant expression object whose value defines the
        volume or area of the compartment. If not specified, the size will be
        fixed at 1.0.

    Attributes
    ----------
    Identical to Parameters (see above).

    Notes
    -----
    The compartments of a model must form a tree via their `parent` attributes
    with a three-dimensional (volume) compartment at the root. A volume
    compartment may have any number of two-dimensional (membrane) compartments
    as its children, but never another volume compartment. A membrane
    compartment may have a single volume compartment as its child, but nothing
    else.

    Examples
    --------
    Compartment('cytosol', dimension=3, size=cyto_vol, parent=ec_membrane)

    """

    def __init__(self, name, parent=None, dimension=3, size=None, _export=True):
        if parent != None and isinstance(parent, Compartment) == False:
            raise Exception("parent must be a predefined Compartment or None")
        #FIXME: check for only ONE "None" parent? i.e. only one compartment can have a parent None?
        if size is not None and not isinstance(size, Parameter) and not \
                (isinstance(size, Expression) and size.is_constant_expression()):
            raise Exception("size must be a parameter or a constant expression"
                            " (or omitted)")
        self.parent = parent
        self.dimension = dimension
        self.size = size
        Component.__init__(self, name, _export)

    def __repr__(self):
        return '%s(name=%s, parent=%s, dimension=%s, size=%s)' % (
            self.__class__.__name__,
            repr(self.name),
            'None' if self.parent is None else self.parent.name,
            repr(self.dimension),
            'None' if self.size is None else self.size.name
        )


class Rule(Component):

    """
    Model component representing a reaction rule.

    Parameters
    ----------
    rule_expression : RuleExpression
        RuleExpression containing the essence of the rule (reactants, products,
        reversibility).
    rate_forward : Union[Parameter,Expression]
        Forward reaction rate constant.
    rate_reverse : Union[Parameter,Expression], optional
        Reverse reaction rate constant (only required for reversible rules).
    delete_molecules : bool, optional
        If True, deleting a Monomer from a species is allowed to fragment the
        species into multiple pieces (if the deleted Monomer was the sole link
        between those pieces). If False (default) then fragmentation is
        disallowed and the rule will not match a reactant species if applying
        the rule would fragment a species.
    move_connected : bool, optional
        If True, a rule that transports a Monomer between compartments will
        co-transport anything connected to that Monomer by a path in the same
        compartment. If False (default), connected Monomers will remain where
        they were.
    total_rate: bool, optional
        If True, the rate is considered to be macroscopic and is not
        multiplied by the number of reactant molecules during simulation.
        If False (default), the rate is multiplied by number of reactant
        molecules.
        Keyword is used by BioNetGen only for simulations using NFsim.
        Keyword is ignored by generate_network command of BioNetGen.

    Attributes
    ----------

    Identical to Parameters (see above), plus the component elements of
    `rule_expression`: reactant_pattern, product_pattern and is_reversible.

    """

    def __init__(self, name, rule_expression, rate_forward, rate_reverse=None,
                 delete_molecules=False, move_connected=False,
                 _export=True, total_rate=False):
        if not isinstance(rule_expression, RuleExpression):
            raise Exception("rule_expression is not a RuleExpression object")
        validate_expr(rate_forward, "forward rate")
        if rule_expression.is_reversible:
            validate_expr(rate_reverse, "reverse rate")
        elif rate_reverse:
            raise ValueError('Reverse rate specified, but rule expression is '
                             'not reversible. Use | instead of >>.')
        self.rule_expression = rule_expression
        self.reactant_pattern = rule_expression.reactant_pattern
        self.product_pattern = rule_expression.product_pattern
        self.is_reversible = rule_expression.is_reversible
        self.rate_forward = rate_forward
        self.rate_reverse = rate_reverse
        self.delete_molecules = delete_molecules
        self.move_connected = move_connected
        self.total_rate = total_rate
        # TODO: ensure all numbered sites are referenced exactly twice within each of reactants and products

        # Check synthesis products are concrete
        if self.is_synth():
            rp = self.reactant_pattern if self.is_reversible else \
                self.product_pattern
            for cp in rp.complex_patterns:
                if not cp.is_concrete():
                    raise ValueError('Product {} of synthesis rule {} is not '
                                     'concrete'.format(cp, name))

        Component.__init__(self, name, _export)

        # Get tags from rule expression
        tags = set()
        for rxn_pat in (rule_expression.reactant_pattern,
                        rule_expression.product_pattern):
            if rxn_pat.complex_patterns:
                for cp in rxn_pat.complex_patterns:
                    if cp is not None:
                        if cp._tag:
                            tags.add(cp._tag)
                        tags.update(mp._tag for mp in cp.monomer_patterns
                                    if mp._tag is not None)

        # Check that tags defined in rates are used in the expression
        tags_rates = (self._check_rate_tags('forward', tags) +
                      self._check_rate_tags('reverse', tags))

        missing = tags.difference(set(tags_rates))
        if missing:
            names = [t.name for t in missing]
            warnings.warn(
                'Rule "{}": Tags {} defined in rule expression but not used in '
                'rates'.format(self.name, ', '.join(names)), UserWarning)

    def _check_rate_tags(self, direction, tags):
        rate = self.rate_forward if direction == 'forward' else \
            self.rate_reverse
        if not isinstance(rate, Expression):
            return []
        tags_rate = rate.tags()
        missing = set(tags_rate).difference(tags)
        if missing:
            names = [t.name for t in missing]
            raise ValueError(
                'Rule "{}": Tag(s) {} defined in {} rate but not in '
                'expression'.format(self.name, ', '.join(names), direction))

        return tags_rate

    def is_synth(self):
        """Return a bool indicating whether this is a synthesis rule."""
        return len(self.reactant_pattern.complex_patterns) == 0 or \
            (self.is_reversible and
             len(self.product_pattern.complex_patterns) == 0)

    def is_deg(self):
        """Return a bool indicating whether this is a degradation rule."""
        return len(self.product_pattern.complex_patterns) == 0 or \
            (self.is_reversible and
             len(self.reactant_pattern.complex_patterns) == 0)

    def __repr__(self):
        ret = '%s(%s, %s, %s' % \
            (self.__class__.__name__, repr(self.name),
             repr(self.rule_expression), self.rate_forward.name)
        if self.is_reversible:
            ret += ', %s' % self.rate_reverse.name
        if self.delete_molecules:
            ret += ', delete_molecules=True'
        if self.move_connected:
            ret += ', move_connected=True'
        ret += ')'
        return ret



def validate_expr(obj, description):
    """Raises an exception if the argument is not an expression."""
    if not isinstance(obj, (Parameter, Expression)):
        description_upperfirst = description[0].upper() + description[1:]
        msg = "%s must be a Parameter or Expression" % description_upperfirst
        raise ExpressionError(msg)

def validate_const_expr(obj, description):
    """Raises an exception if the argument is not a constant expression."""
    validate_expr(obj, description)
    if isinstance(obj, Expression) and not obj.is_constant_expression():
        description_upperfirst = description[0].upper() + description[1:]
        msg = ("%s must be a Parameter or constant Expression" %
               description_upperfirst)
        raise ConstantExpressionError(msg)


class Observable(Component, Symbol):

    """
    Model component representing a linear combination of species.

    Observables are useful in correlating model simulation results with
    experimental measurements. For example, an observable for "A()" will report
    on the total number of copies of Monomer A, regardless of what it's bound to
    or the state of its sites. "A(y='P')" would report on all instances of A
    with site 'y' in state 'P'.

    Parameters
    ----------
    reaction_pattern : ReactionPattern
        The list of ComplexPatterns to match.
    match : 'species' or 'molecules'
        Whether to match entire species ('species') or individual fragments
        ('molecules'). Default is 'molecules'.

    Attributes
    ----------
    reaction_pattern : ReactionPattern
        See Parameters.
    match : 'species' or 'molecules'
        See Parameters.
    species : list of integers
        List of species indexes for species matching the pattern.
    coefficients : list of integers
        List of coefficients by which each species amount is to be multiplied to
        correct for multiple pattern matches within a species.

    Notes
    -----
    ReactionPattern is used here as a container for a list of ComplexPatterns,
    solely so users could utilize the ComplexPattern '+' operator overload as
    syntactic sugar. There are no actual "reaction" semantics in this context.

    """

    def __new__(cls, name, reaction_pattern, match='molecules', _export=True):
        return super(Observable, cls).__new__(cls, name)

    def __getnewargs__(self):
        return self.name, self.reaction_pattern, self.match, False

    def __init__(self, name, reaction_pattern, match='molecules', _export=True):
        try:
            reaction_pattern = as_reaction_pattern(reaction_pattern)
        except InvalidReactionPatternException as e:
            raise type(e)("Observable pattern does not look like a ReactionPattern")
        if match not in ('molecules', 'species'):
            raise ValueError("Match must be 'molecules' or 'species'")
        Component.__init__(self, name, _export)
        self.reaction_pattern = reaction_pattern
        self.match = match
        self.species = []
        self.coefficients = []

    def expand_obs(self):
        """ Expand observables in terms of species and coefficients """
        return sympy.Add(*[a * b for a, b in zip(
            self.coefficients,
            [sympy.Symbol('__s%d' % sp_id) for sp_id in self.species]
        )])

    def __repr__(self):
        ret = '%s(%s, %s' % (self.__class__.__name__, repr(self.name),
                              repr(self.reaction_pattern))
        if self.match != 'molecules':
            ret += ', match=%s' % repr(self.match)
        ret += ')'
        return ret

    def __str__(self):
        return repr(self)

    def __call__(self, tag):
        if not isinstance(tag, Tag):
            raise ValueError('Observables are only callable with a Tag '
                             'instance, for use within local Expressions')

        return sympy.Function(self.name)(tag)


class Expression(Component, Symbol):

    """
    Model component representing a symbolic expression of other variables.

    Parameters
    ----------
    expr : sympy.Expr
        Symbolic expression.

    Attributes
    ----------
    expr : sympy.Expr
        See Parameters.

    """

    def __new__(cls, name, expr, _export=True):
        return super(Expression, cls).__new__(cls, name)

    def __getnewargs__(self):
        return self.name, self.expr, False

    def __init__(self, name, expr, _export=True):
        if not isinstance(expr, sympy.Expr):
            raise ValueError('An Expression can only be created from a '
                             'sympy.Expr object')
        self.expr = expr
        Component.__init__(self, name, _export)

    def expand_expr(self, expand_observables=False):
        """Return expr rewritten in terms of terminal symbols only."""
        subs = []
        for a in self.expr.atoms():
            if isinstance(a, Expression):
                subs.append((a, a.expand_expr(
                    expand_observables=expand_observables)))
            elif expand_observables and isinstance(a, Observable):
                subs.append((a, a.expand_obs()))
        return self.expr.subs(subs)

    def is_constant_expression(self):
        """Return True if all terminal symbols are Parameters or numbers."""
        return all(isinstance(a, Parameter) or
                   (isinstance(a, Expression) and a.is_constant_expression()) or
                   isinstance(a, sympy.Number)
                   for a in self.expr.atoms())

    def get_value(self):
        # Use parameter and expression values for evaluation
        subs = {}
        for a in self.expr.atoms():
            if isinstance(a, Parameter):
                subs[a] = a.value
            elif isinstance(a, Expression) and a.is_constant_expression():
                subs[a] = a.get_value()
        return self.expr.xreplace(subs)

    @property
    def is_local(self):
        return len(self.expr.atoms(Tag)) > 0

    def tags(self):
        return sorted(self.expr.atoms(Tag), key=lambda tag: tag.name)

    def __repr__(self):
        if isinstance(self.expr, (Parameter, Expression)):
            expr_repr = self.expr.name
        else:
            expr_repr = repr(self.expr)
        ret = '%s(%s, %s)' % (self.__class__.__name__, repr(self.name),
                              expr_repr)
        return ret

    def __str__(self):
        return repr(self)

    def __call__(self, tag):
        if not isinstance(tag, Tag):
            raise ValueError('Expressions are only callable with a Tag '
                             'instance, for use within local Expressions')

        return sympy.Function(self.name)(tag)


class Tag(Component, Symbol):
    """Tag for labelling MonomerPatterns and ComplexPatterns"""
    def __new__(cls, name, _export=True):
        return super(Tag, cls).__new__(cls, name)

    def __getnewargs__(self):
        return self.name, False

    def __init__(self, name, _export=True):
        Component.__init__(self, name, _export)

    def __matmul__(self, other):
        if not isinstance(other, MonomerPattern):
            return NotImplemented

        if other._tag:
            raise TagAlreadySpecifiedError()

        new_mp = other()
        new_mp._tag = self
        return new_mp

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, repr(self.name))


class Initial(object):
    """
    An initial condition for a species.

    An initial condition is made up of a species, its amount or concentration,
    and whether it is to be held fixed during a simulation.

    Species patterns must satisfy all of the following:
    * Able to be cast as a ComplexPattern
    * Concrete (see ComplexPattern.is_concrete)
    * Distinct from any existing initial condition pattern
    * match_once is False (nonsensical in this context)

    Parameters
    ----------
    pattern : ComplexPattern
        A concrete pattern defining the species to initialize.
    value : Parameter or Expression Amount of the species the model will start
        with. If an Expression is used, it must evaluate to a constant (can't
        reference any Observables).
    fixed : bool
        Whether or not the species should be held fixed (never consumed).

    Attributes
    ----------
    Identical to Parameters (see above).

    """

    def __init__(self, pattern, value, fixed=False, _export=True):
        try:
            pattern = as_complex_pattern(pattern)
        except InvalidComplexPatternException as e:
            raise InvalidInitialConditionError("Not a ComplexPattern")
        if not pattern.is_concrete():
            raise InvalidInitialConditionError("Pattern not concrete")
        if pattern.match_once:
            raise InvalidInitialConditionError("MatchOnce not allowed here")
        validate_const_expr(value, "initial condition value")
        self.pattern = pattern
        self.value = value
        self.fixed = fixed
        self._export = _export
        if self._export:
            SelfExporter.add_initial(self)

    def __repr__(self):
        ret = '%s(%s, %s' % (self.__class__.__name__, repr(self.pattern),
                             self.value.name)
        if self.fixed:
            ret += ', fixed=True'
        ret += ')'
        return ret


class Model(object):

    """
    A rule-based model containing monomers, rules, compartments and parameters.

    Parameters
    ----------
    name : string, optional
        Name of the model. If not specified, will be set to the name of the file
        from which the constructor was called (with the .py extension stripped).
    base : Model, optional
        If specified, the model will begin as a copy of `base`. This can be used
        to achieve a simple sort of model extension and enhancement.

    Attributes
    ----------
    name : string
        Name of the model. See Parameter section above.
    base : Model or None
        See Parameter section above.
    monomers, compartments, parameters, rules, observables : ComponentSet
        The Component objects which make up the model.
    initials : list of Initial
        Specifies which species are present in the model's starting
        state (t=0) and how much there is of each one.
    initial_conditions : list of tuple of (ComplexPattern, Parameter)
        The old representation of initial conditions, deprecated in favor of
        `initials`.
    species : list of ComplexPattern
        List of all complexes which can be produced by the model, starting from
        the initial conditions and successively applying the rules. Each 
        ComplexPattern is concrete.
    reactions : list of dict
        Structures describing each possible unidirectional reaction that can be
        produced by the model. Each structure stores the name of the rule that
        generated the reaction ('rule'), the mathematical expression for the
        rate of the reaction ('rate'), tuples of species indexes for the
        reactants and products ('reactants', 'products'), and a bool indicating
        whether the reaction is the reverse component of a bidirectional
        reaction ('reverse').
    reactions_bidirectional : list of dict
        Similar to `reactions` but with only one entry for each bidirectional
        reaction. The fields are identical except 'reverse' is replaced by
        'reversible', a bool indicating whether the reaction is reversible. The
        'rate' is the forward rate minus the reverse rate.
    annotations : list of Annotation
        Structured annotations of model components. See the Annotation class for
        details.

    """

    _component_types = (Monomer, Compartment, Parameter, Rule, Observable,
                        Expression, Tag)

    def __init__(self, name=None, base=None, _export=True):
        self.name = name
        self.base = base
        self._export = _export
        self.monomers = ComponentSet()
        self.compartments = ComponentSet()
        self.parameters = ComponentSet()
        self.rules = ComponentSet()
        self.observables = ComponentSet()
        self.expressions = ComponentSet()
        self.tags = ComponentSet()
        self.initials = []
        self.annotations = []
        self._odes = OdeView(self)
        self._initial_conditions = InitialConditionsView(self)
        self.reset_equations()
        #####
        self.diffusivities = []
        #####
        if self._export:
            SelfExporter.export(self)
        if self.base is not None:
            if not isinstance(self.base, Model):
                raise ValueError("base must be a Model")
            model_copy = copy.deepcopy(self.base)
            for component in model_copy.all_components():
                self.add_component(component)
                component._do_export()
            self.initials = model_copy.initials
            self.annotations = model_copy.annotations

    def __getstate__(self):
        state = self.__dict__.copy()
        # The stoichiometry matrix, as a numpy array, is problematic to pickle
        # in a cross-Python-version-compatible way. Since it's regenerated on
        # demand anyway, we can just clear it here.
        state['_stoichiometry_matrix'] = None
        return state

    def __setstate__(self, state):
        # restore the 'model' weakrefs on all components
        self.__dict__.update(state)
        for c in self.all_components():
            c.model = weakref.ref(self)

    def reload(self):
        """
        Reload a model after its source files have been edited.

        This method does not yet reload the model contents in-place, rather it
        returns a new model object. Thus the correct usage is ``model =
        model.reload()``.

        If the model script imports any modules, these will not be reloaded. Use
        python's reload() function to reload them.

        """
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

    @property
    def modules(self):
        """
        Return the set of Python modules where Components are defined

        Returns
        -------
        list
            List of module names where model Components are defined

        Examples
        --------

        >>> from pysb.examples.earm_1_0 import model
        >>> 'pysb.examples.earm_1_0' in model.modules
        True
        """
        all_components = self.components
        if not all_components:
            return []
        return sorted(set.union(*[set(c._modules) for c in all_components]))

    def all_component_sets(self):
        """Return a list of all ComponentSet objects."""
        set_names = [t.__name__.lower() + 's' for t in Model._component_types]
        sets = [getattr(self, name) for name in set_names]
        return sets

    def all_components(self):
        """Return a ComponentSet containing all components in the model."""
        cset_all = ComponentSet()
        for cset in self.all_component_sets():
            cset_all |= cset
        return cset_all

    @property
    def components(self):
        return self.all_components()

    def parameters_all(self):
        """Return a ComponentSet of all parameters and derived parameters."""
        return self.parameters | self._derived_parameters

    def parameters_rules(self):
        """Return a ComponentSet of the parameters used in rules."""
        # rate_reverse is None for irreversible rules, so we'll need to filter those out
        cset = ComponentSet(p for r in self.rules for p in (r.rate_forward, r.rate_reverse)
                            if p is not None)
        # intersect with original parameter list to retain ordering
        return self.parameters & cset

    def parameters_initial_conditions(self):
        """Return a ComponentSet of initial condition parameters."""
        cset = ComponentSet(ic.value for ic in self.initials)
        # intersect with original parameter list to retain ordering
        return self.parameters & cset

    def parameters_compartments(self):
        """Return a ComponentSet of compartment size parameters."""
        cset = ComponentSet(c.size for c in self.compartments)
        # intersect with original parameter list to retain ordering
        return self.parameters & cset

    def parameters_expressions(self):
        """Return a ComponentSet of the parameters used in expressions."""
        cset = ComponentSet()
        for expr in self.expressions:
            for sym in expr.expand_expr().free_symbols:
                if sym in self.parameters:
                    cset.add(sym)
        # intersect with original parameter list to retain ordering
        return self.parameters & cset

    def parameters_unused(self):
        """Return a ComponentSet of unused parameters."""
        cset_used = (self.parameters_rules() | self.parameters_initial_conditions() |
                     self.parameters_compartments() | self.parameters_expressions())
        return self.parameters - cset_used

    def expressions_constant(self, include_derived=False):
        """Return a ComponentSet of constant expressions."""
        expressions = self.expressions
        if include_derived:
            expressions = expressions | self._derived_expressions
        cset = ComponentSet(e for e in expressions if e.is_constant_expression())
        return cset

    def expressions_dynamic(self, include_local=True, include_derived=False):
        """Return a ComponentSet of non-constant expressions."""
        expressions = self.expressions
        if include_derived:
            expressions = expressions | self._derived_expressions
        cset = expressions - self.expressions_constant(include_derived)
        if not include_local:
            cset = ComponentSet(e for e in cset if not e.is_local)
        return cset

    @property
    def odes(self):
        """Return sympy Expressions for the time derivative of each species."""
        return self._odes

    @property
    def stoichiometry_matrix(self):
        """Return the stoichiometry matrix for the reaction network."""
        if self._stoichiometry_matrix is None:
            shape = (len(self.species), len(self.reactions))
            sm = scipy.sparse.lil_matrix(shape, dtype='int')
            for i, reaction in enumerate(self.reactions):
                for r in reaction['reactants']:
                    sm[r, i] -= 1
                for p in reaction['products']:
                    sm[p, i] += 1
            fixed = [i for i, ic in enumerate(self.initials) if ic.fixed]
            sm[fixed, :] = 0
            self._stoichiometry_matrix = sm.tocsr()
        return self._stoichiometry_matrix

    def add_component(self, other):
        """Add a component to the model."""
        # We have a container for each type of component. This code determines
        # the right one based on the class of the object being added.
        for t, cset in zip(Model._component_types, self.all_component_sets()):
            if isinstance(other, t):
                cset.add(other)
                other.model = weakref.ref(self)
                break
        else:
            raise Exception("Tried to add component of unknown type '%s' to "
                            "model" % type(other))

    def add_annotation(self, annotation):
        """Add an annotation to the model."""
        self.annotations.append(annotation)

    def get_annotations(self, subject):
        """Return all annotations for the given subject."""
        annotations = []
        for a in self.annotations:
            if a.subject is subject:
                annotations.append(a)
        return annotations

    def _rename_component(self, component, new_name):
        """
        Change a component's name.

        This has to be done through the Model because the ComponentSet needs to
        be updated as well as the component's `name` field.

        """
        for cset in self.all_component_sets():
            if component in cset:
                cset.rename(component, new_name)

    def add_initial(self, initial):
        if initial in self.initials:
            return
        if any(
            initial.pattern.is_equivalent_to(other.pattern)
            for other in self.initials
        ):
            raise InvalidInitialConditionError("Duplicate species")
        self.initials.append(initial)

    def initial(self, pattern, value, fixed=False):
        """Add an initial condition.

        This method is deprecated. Instead, create an Initial object
        and pass it to add_initial.

        """
        warnings.warn(
            'Model.initial will be removed in a future version. Instead,'
            ' create an Initial object and pass it to Model.add_initial.',
            DeprecationWarning
        )
        self.add_initial(Initial(pattern, value, fixed, _export=False))

    @property
    def initial_conditions(self):
        warnings.warn(
            'Model.initial_conditions will be removed in a future version.'
            ' Instead, you can get a list of Initial objects with'
            ' Model.initials.', DeprecationWarning
        )
        return self._initial_conditions

    def update_initial_condition_pattern(self, before_pattern, after_pattern):
        """
        Update the pattern associated with an initial condition.

        Leaves the Parameter object associated with the initial condition
        unchanged while modifying the pattern associated with that condition.
        For example this is useful for changing the state of a site on a
        monomer or complex associated with an initial condition without having
        to create an independent initial condition, and parameter, associated
        with that alternative state.

        Parameters
        ----------
        before_pattern : ComplexPattern
            The concrete pattern specifying the (already existing) initial
            condition. If the model does not contain an initial condition
            for the pattern, a ValueError is raised.
        after_pattern : ComplexPattern
            The concrete pattern specifying the new pattern to use to replace
            before_pattern.
        """

        before_pattern = as_complex_pattern(before_pattern)

        # Get the initial condition index
        ic_index_list = [
            i for i, ic in enumerate(self.initials)
            if ic.pattern.is_equivalent_to(before_pattern)
        ]

        # If the initial condition to replace is not found, raise an error
        if not ic_index_list:
            raise ValueError("No initial condition found for pattern %s" %
                             before_pattern)

        # If more than one matching initial condition is found, raise an
        # error (this should never happen, because duplicate initial conditions
        # are not allowed to be created)
        assert len(ic_index_list) == 1

        # Replace the pattern in the initial condition
        initial_index = ic_index_list[0]
        self.initials[initial_index].pattern = after_pattern

    def get_species_index(self, complex_pattern):
        """
        Return the index of a species.

        Parameters
        ----------
        complex_pattern : ComplexPattern
            A concrete pattern specifying the species to find.

        """
        # FIXME I don't even want to think about the inefficiency of this, but at least it works
        try:
            return next((i for i, s_cp in enumerate(self.species) if s_cp.is_equivalent_to(complex_pattern)))
        except StopIteration:
            return None

    def has_synth_deg(self):
        """Return true if model uses synthesis or degradation reactions."""
        return any(r.is_synth() or r.is_deg() for r in self.rules)

    def enable_synth_deg(self):
        """Add components needed to support synthesis and degradation rules."""
        warnings.warn('This function is no longer needed, and no longer has '
                      'any effect.', DeprecationWarning)

    def reset_equations(self):
        """Clear out fields generated by bng.generate_equations or the like."""
        self.species = []
        self.reactions = []
        self.reactions_bidirectional = []
        self._stoichiometry_matrix = None
        self._derived_parameters = ComponentSet()
        self._derived_expressions = ComponentSet()
        for obs in self.observables:
            obs.species = []
            obs.coefficients = []

    def __repr__(self):
        return ("<%s '%s' (monomers: %d, rules: %d, parameters: %d, "
                "expressions: %d, compartments: %d) at 0x%x>" %
                (self.__class__.__name__, self.name,
                 len(self.monomers), len(self.rules), len(self.parameters),
                 len(self.expressions), len(self.compartments), id(self)))



class InvalidComplexPatternException(Exception):
    """Expression can not be cast as a ComplexPattern."""
    pass

class InvalidReactionPatternException(Exception):
    """Expression can not be cast as a ReactionPattern."""
    pass

class InvalidReversibleSynthesisDegradationRule(Exception):
    """Synthesis or degradation rule defined as reversible."""
    def __init__(self):
        Exception.__init__(self, "Synthesis and degradation rules may not be"
                           "reversible.")

class ExpressionError(ValueError):
    """Expected an Expression but got something else."""
    pass

class ConstantExpressionError(ValueError):
    """Expected a constant Expression but got something else."""
    pass

class ModelExistsWarning(UserWarning):
    """A second model was declared in a module that already contains one."""
    pass

class SymbolExistsWarning(UserWarning):
    """A component declaration or rename overwrote an existing symbol."""
    pass

class InvalidComponentNameError(ValueError):
    """Inappropriate component name."""
    def __init__(self, name):
        ValueError.__init__(self, "Not a valid component name: '%s'" % name)

class InvalidInitialConditionError(ValueError):
    """Invalid initial condition pattern."""

class DuplicateMonomerError(ValueError):
    pass

class DuplicateSiteError(ValueError):
    pass

class UnknownSiteError(ValueError):
    pass

class CompartmentAlreadySpecifiedError(ValueError):
    pass


class TagAlreadySpecifiedError(ValueError):
    pass


class ModelNotDefinedError(RuntimeError):
    """SelfExporter method was called before a model was defined."""
    def __init__(self):
        super(RuntimeError, self).__init__(
            "A Model must be declared before declaring any model components"
        )


class ComponentSet(Set, Mapping, Sequence):
    """
    An add-and-read-only container for storing model Components.

    It behaves mostly like an ordered set, but components can also be retrieved
    by name *or* index by using the [] operator (like a combination of a dict
    and a list). Components cannot be removed or replaced, but they can be
    renamed. Iteration returns the component objects.

    Parameters
    ----------
    iterable : iterable of Components, optional
        Initial contents of the set.

    """

    # The implementation is based on a list instead of a linked list (as
    # OrderedSet is), since we only allow add and retrieve, not delete.

    def __init__(self, iterable=None):
        self._elements = []
        self._map = {}
        self._index_map = {}
        if iterable is not None:
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
                raise ComponentDuplicateNameError(
                    "Tried to add a component with a duplicate name: %s"
                    % c.name)
            self._elements.append(c)
            self._map[c.name] = c
            self._index_map[c.name] = len(self._elements) - 1

    def __getitem__(self, key):
        # Must support both Sequence and Mapping behavior. This means
        # stringified integer Mapping keys (like "0") are forbidden, but since
        # all Component names must be valid Python identifiers, integers are
        # ruled out anyway.
        if isinstance(key, (int, slice)):
            return self._elements[key]
        else:
            return self._map[key]

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError("Model has no component '%s'" % name)

    def __setstate__(self, state):
        self.__dict__ = state

    def __dir__(self):
        return self.keys()

    def get(self, key, default=None):
        if isinstance(key, int):
            raise ValueError("get is undefined for integer arguments, use []"
                             "instead")
        try:
            return self[key]
        except KeyError:
            return default

    def filter(self, filter_predicate):
        """
        Filter a ComponentSet using a predicate or set of predicates

        Parameters
        ----------
        filter_predicate: callable or pysb.pattern.FilterPredicate
            A predicate (condition) to test each Component in the
            ComponentSet against. This can either be an anonymous "lambda"
            function or a subclass of pysb.pattern.FilterPredicate. For
            lambda functions, the argument is a single Component and return
            value is a boolean indicating a match or not.

        Returns
        -------
        ComponentSet
            A ComponentSet containing Components matching all of the
            supplied filters

        Examples
        --------

        >>> from pysb.examples.earm_1_0 import model
        >>> from pysb.pattern import Name, Pattern, Module, Function
        >>> m = model.monomers

        Find parameters exactly equal to 10000:

        >>> model.parameters.filter(lambda c: c.value == 1e4)  \
            # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Parameter('pC3_0', 10000.0),
         Parameter('pC6_0', 10000.0),
        ])

        Find rules with a forward rate < 1e-8, using a custom function:

        >>> model.rules.filter(lambda c: c.rate_forward.value < 1e-8) \
            # doctest: +NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('bind_pC3_Apop', Apop(b=None) + pC3(b=None) | Apop(b=1) %
                pC3(b=1), kf25, kr25),
        ])

        We can also use some built in predicates for more complex matching
        scenarios, including combining multiple predicates.

        Find rules with a name beginning with "inhibit" that contain cSmac:

        >>> model.rules.filter(Name('^inhibit') & Pattern(m.cSmac())) \
            # doctest: +NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('inhibit_cSmac_by_XIAP', cSmac(b=None) + XIAP(b=None) |
                cSmac(b=1) % XIAP(b=1), kf28, kr28),
        ])

        Find rules with any form of Bax (i.e. Bax, aBax, mBax):

        >>> model.rules.filter(Pattern(m.Bax) | Pattern(m.aBax) | \
                Pattern(m.MBax)) # doctest: +NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('bind_Bax_tBid', tBid(b=None) + Bax(b=None) |
              tBid(b=1) % Bax(b=1), kf12, kr12),
         Rule('produce_aBax_via_tBid', tBid(b=1) % Bax(b=1) >>
              tBid(b=None) + aBax(b=None), kc12),
         Rule('transloc_MBax_aBax', aBax(b=None) |
              MBax(b=None), kf13, kr13),
         Rule('inhibit_MBax_by_Bcl2', MBax(b=None) + Bcl2(b=None) |
              MBax(b=1) % Bcl2(b=1), kf14, kr14),
         Rule('dimerize_MBax_to_Bax2', MBax(b=None) + MBax(b=None) |
              Bax2(b=None), kf15, kr15),
         ])

        Count the number of parameter that don't start with kf (note the ~
        negation operator):

        >>> len(model.parameters.filter(~Name('^kf')))
        60

        Get components not defined in this module (file). In this case,
        everything is defined in one file, but for multi-file models this
        becomes more useful:

        >>> model.components.filter(~Module('^pysb.examples.earm_1_0$'))
        ComponentSet([
         ])

        Count the number of rules defined in the 'catalyze' function:

        >>> len(model.rules.filter(Function('^catalyze$')))
        24

        """
        return ComponentSet(filter(filter_predicate, self))

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
        return list(zip(self.keys(), self))

    def index(self, c):
        # We can implement this in O(1) ourselves, whereas the Sequence mixin
        # implements it in O(n).
        if not c in self:
            raise ValueError("%s is not in ComponentSet" % c)
        return self._index_map[c.name]

    def __and__(self, other):
        # We reimplement this because collections.Set's __and__ mixin iterates
        # over other, not self. That implementation ends up retaining the
        # ordering of other, but we'd like to keep the ordering of self instead.
        # We require other to be a ComponentSet too so we know it will support
        # "in" efficiently.
        if not isinstance(other, ComponentSet):
            return Set.__and__(self, other)
        return ComponentSet(value for value in self if value in other)

    def __rand__(self, other):
        return self.__and__(other)

    def __ror__(self, other):
        return self.__or__(other)

    def __rxor__(self, other):
        return self.__xor__(other)

    def __repr__(self):
        return 'ComponentSet([\n' + \
            ''.join(' %s,\n' % repr(x) for x in self) + \
            ' ])'

    def rename(self, c, new_name):
        """Change the name of component `c` to `new_name`."""
        for m in self._map, self._index_map:
            m[new_name] = m[c.name]
            del m[c.name]


class OdeView(Sequence):
    """Compatibility shim for the Model.odes property."""

    # This is necessarily coupled pretty tightly with Model. Note that we
    # faithfully emulate the detail of the original implementation in which odes
    # is an empty list before the equation generation process is run (and after
    # reset_equations is called). Now the "empty" condition is when species is
    # empty.

    def __init__(self, model):
        self.model = model

    def __getitem__(self, key):
        if not self.model.species:
            raise IndexError('list index out of range')
        if isinstance(key, slice):
            return [self[k] for k in range(*key.indices(len(self)))]
        sr = self.model.stoichiometry_matrix[key]
        terms = [sympy.Mul(self.model.reactions[i]['rate'], v, evaluate=False)
                 for i, v in zip(sr.indices, sr.data)]
        return sympy.Add(*terms, evaluate=False)

    def __len__(self):
        return len(self.model.species)


class InitialConditionsView(Sequence):
    """Compatibility shim for the Model.initial_conditions property."""

    def __init__(self, model):
        self.model = model

    def __getitem__(self, key):
        initial = self.model.initials[key]
        return (initial.pattern, initial.value)

    def __len__(self):
        return len(self.model.initials)


class ComponentDuplicateNameError(ValueError):
    """A component was added with the same name as an existing one."""
    pass


def extract_site_conditions(conditions=None, **kwargs):
    """Parse MonomerPattern/ComplexPattern site conditions."""
    # enforce site conditions as kwargs or a dict but not both
    if conditions and kwargs:
        raise RedundantSiteConditionsError()
    # handle normal cases
    elif conditions:
        site_conditions = conditions.copy()
    else:
        site_conditions = kwargs
    return site_conditions


class RedundantSiteConditionsError(ValueError):
    """Both conditions dict and kwargs both passed to create pattern."""
    def __init__(self):
        ValueError.__init__(
            self,
            ("Site conditions may be specified as EITHER keyword arguments "
             "OR a single dict"))


class DanglingBondError(ValueError):
    pass


class ReusedBondError(ValueError):
    pass

# Some light infrastructure for defining symbols that act like "keywords", i.e.
# they are immutable singletons that stringify to their own name. Regular old
# classes almost fit the bill, except that their __str__ method prepends the
# complete module hierarchy to the base class name. The KeywordMeta class here
# implements an alternate __str__ method which just returns the base name.

class KeywordMeta(type):
    def __repr__(cls):
        return cls.__name__
    def __str__(cls):
        return repr(cls)


# Define Keyword class with KeywordMeta metaclass in a Python 2 and 3
# compatible way
class Keyword(KeywordMeta("KeywordMetaBase", (object, ), {})):
    pass

# The keywords.

class ANY(Keyword):
    """Site must have a bond, but identity of binding partner is irrelevant.

    Use ANY in a MonomerPattern site_conditions dict to indicate that a site
    must have a bond without specifying what the binding partner should be.

    Equivalent to the "+" bond modifier in BNG."""
    pass

class WILD(Keyword):
    """Site may be bound or unbound.

    Use WILD as part of a (state, WILD) tuple in a MonomerPattern
    site_conditions dict to indicate that a site must have the given state,
    irrespective of the presence or absence of a bond. (Specifying only the
    state implies there must not be a bond). A bare WILD in a site_conditions
    dict is also permissible, but as this has the same meaning as the much
    simpler option of leaving the given site out of the dict entirely, this
    usage is deprecated.

    Equivalent to the "?" bond modifier in BNG."""
    pass


warnings.simplefilter('always', ModelExistsWarning)
warnings.simplefilter('always', SymbolExistsWarning)
