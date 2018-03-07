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

try:
    reload
except NameError:
    from imp import reload
try:
    basestring
except NameError:
    # Under Python 3, do not pretend that bytes are a valid string
    basestring = str
    long = int

def Initial(*args):
    """Declare an initial condition (see Model.initial)."""
    return SelfExporter.default_model.initial(*args)

def MatchOnce(pattern):
    """Make a ComplexPattern match-once."""
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
            if SelfExporter.default_model == None:
                raise Exception("A Model must be declared before declaring any model components")
            SelfExporter.default_model.add_component(obj)

        # load obj into target namespace under obj.name
        if export_name in SelfExporter.target_globals:
            warnings.warn("'%s' already defined" % (export_name), SymbolExistsWarning, stacklevel)
        SelfExporter.target_globals[export_name] = obj

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

    def __init__(self, name, _export=True):
        if not re.match(r'[_a-z][_a-z0-9]*\Z', name, re.IGNORECASE):
            raise InvalidComponentNameError(name)
        self.name = name
        self.model = None  # to be set in Model.add_component
        self._export = _export
        if self._export:
            self._do_export()

    def __getstate__(self):
        # clear the weakref to parent model (restored in Model.__setstate__)
        state = self.__dict__.copy()
        del state['model']
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

    """

    def __init__(self, name, sites=None, site_states=None, _export=True):
        Component.__init__(self, name, _export)

        # Create default empty containers.
        if sites is None:
            sites = []
        if site_states is None:
            site_states = {}

        # ensure sites is some kind of list (presumably of strings) but not a
        # string itself
        if not isinstance(sites, collections.Iterable) or \
               isinstance(sites, basestring):
            raise ValueError("sites must be a list of strings")

        # ensure no duplicate sites
        sites_seen = {}
        for site in sites:
            sites_seen.setdefault(site, 0)
            sites_seen[site] += 1
        sites_dup = [site for site, count in sites_seen.items() if count > 1]
        if sites_dup:
            raise Exception("Duplicate sites specified: " + str(sites_dup))

        # ensure site_states keys are all known sites
        unknown_sites = [site for site in site_states if not site in sites_seen]
        if unknown_sites:
            raise Exception("Unknown sites in site_states: " +
                            str(unknown_sites))
        # ensure site_states values are all strings
        invalid_sites = [site for (site, states) in site_states.items()
                              if not all([isinstance(s, basestring)
                                          for s in states])]
        if invalid_sites:
            raise Exception("Non-string state values in site_states for "
                            "sites: " + str(invalid_sites))

        self.sites = list(sites)
        self.site_states = site_states

    def __call__(self, conditions=None, **kwargs):
        """
        Return a MonomerPattern object based on this Monomer.

        See the Notes section of this class's documentation for details.

        Parameters
        ----------
        conditions : dict, optional
            See MonomerPattern.site_conditions.
        **kwargs : dict
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

        # ensure each value is one of: None, integer, list of integers, string,
        # (string,integer), (string,WILD), ANY, WILD
        invalid_sites = []
        for (site, state) in site_conditions.items():
            # pass through to next iteration if state type is ok
            if state == None:
                continue
            elif isinstance(state, int):
                continue
            elif isinstance(state, list) and \
                 all(isinstance(s, int) for s in state):
                continue
            elif isinstance(state, basestring):
                _check_state(monomer, site, state)
                continue
            elif isinstance(state, tuple) and \
                 isinstance(state[0], basestring) and \
                 (isinstance(state[1], int) or state[1] is WILD or \
                  state[1] is ANY):
                _check_state(monomer, site, state[0])
                continue
            elif state is ANY:
                continue
            elif state is WILD:
                continue
            invalid_sites.append(site)
        if invalid_sites:
            raise Exception("Invalid state value for sites: " +
                            '; '.join(['%s=%s' % (s,str(site_conditions[s]))
                                       for s in invalid_sites]))

        # ensure compartment is a Compartment
        if compartment and not isinstance(compartment, Compartment):
            raise Exception("compartment is not a Compartment object")

        self.monomer = monomer
        self.site_conditions = site_conditions
        self.compartment = compartment
        self._graph = None

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
        if len(self.site_conditions) != len(self.monomer.sites):
            return False
        for site_name, site_val in self.site_conditions.items():
            if isinstance(site_val, basestring):
                site_state = site_val
                site_bond = None
            elif isinstance(site_val, collections.Iterable):
                site_state, site_bond = site_val
            elif isinstance(site_val, int):
                site_bond = site_val
                site_state = None
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
        return MonomerPattern(self.monomer, site_conditions, self.compartment)

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

    def __ne__(self, other):
        warnings.warn("'<>' for reversible rules will be removed in a future "
                      "version of PySB. Use '|' instead.",
                      PendingDeprecationWarning,
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

    def __repr__(self):
        value = '%s(' % self.monomer.name
        value += ', '.join([
                k + '=' + repr(self.site_conditions[k])
                for k in self.monomer.sites
                if k in self.site_conditions
                ])
        value += ')'
        if self.compartment is not None:
            value += ' ** ' + self.compartment.name
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

        self.monomer_patterns = monomer_patterns
        self.compartment = compartment
        self.match_once = match_once
        self._graph = None

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

        for mp in self.monomer_patterns:
            mon_node_id = next(node_count)
            g.add_node(mon_node_id, id=mp.monomer)
            if mp.compartment or self.compartment:
                cpt_node_id = add_or_get_compartment_node(mp.compartment or
                                                          self.compartment)
                g.add_edge(mon_node_id, cpt_node_id)

            for site, state_or_bond in mp.site_conditions.items():
                mon_site_id = next(node_count)
                g.add_node(mon_site_id, id=site)
                g.add_edge(mon_node_id, mon_site_id)
                state = None
                bond_num = None
                if state_or_bond is WILD:
                    continue
                elif isinstance(state_or_bond, basestring):
                    state = state_or_bond
                elif isinstance(state_or_bond, collections.Iterable) and len(
                        state_or_bond) == 2:
                    state = state_or_bond[0]
                    bond_num = state_or_bond[1]
                elif isinstance(state_or_bond, int):
                    bond_num = state_or_bond

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
        return ComplexPattern([mp() for mp in self.monomer_patterns], self.compartment, self.match_once)

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

    def __ne__(self, other):
        warnings.warn("'<>' for reversible rules will be removed in a future "
                      "version of PySB. Use '|' instead.",
                      PendingDeprecationWarning,
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

    def __repr__(self):
        ret = ' % '.join([repr(p) for p in self.monomer_patterns])
        if self.compartment is not None:
            ret = '(%s) ** %s' % (ret, self.compartment.name)
        if self.match_once:
            ret = 'MatchOnce(%s)' % ret
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
                      PendingDeprecationWarning,
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


class Parameter(Component, sympy.Symbol):

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

    Attributes
    ----------
    Identical to Parameters (see above).

    """

    def __new__(cls, name, value=0.0, _export=True):
        return super(sympy.Symbol, cls).__new__(cls, name)

    def __getnewargs__(self):
        return (self.name, self.value, False)

    def __init__(self, name, value=0.0, _export=True):
        Component.__init__(self, name, _export)
        self.value = float(value)
    
    def get_value(self):
        return self.value
    
    # This is needed to make sympy's evalf machinery treat this class like a
    # Symbol.
    @property
    def func(self):
        return sympy.Symbol

    def __repr__(self):
        return  '%s(%s, %s)' % (self.__class__.__name__, repr(self.name), repr(self.value))

    def __str__(self):
        return  repr(self)



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
    size : Parameter, optional
        A parameter object whose value defines the volume or area of the
        compartment. If not specified, the size will be fixed at 1.0.

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
        Component.__init__(self, name, _export)
        if parent != None and isinstance(parent, Compartment) == False:
            raise Exception("parent must be a predefined Compartment or None")
        #FIXME: check for only ONE "None" parent? i.e. only one compartment can have a parent None?
        if size is not None and not isinstance(size, Parameter):
            raise Exception("size must be a parameter (or omitted)")
        self.parent = parent
        self.dimension = dimension
        self.size = size

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
    rate_forward : Parameter
        Forward reaction rate constant.
    rate_reverse : Parameter, optional
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

    Attributes
    ----------

    Identical to Parameters (see above), plus the component elements of
    `rule_expression`: reactant_pattern, product_pattern and is_reversible.

    """

    def __init__(self, name, rule_expression, rate_forward, rate_reverse=None,
                 delete_molecules=False, move_connected=False,
                 _export=True):
        Component.__init__(self, name, _export)
        if not isinstance(rule_expression, RuleExpression):
            raise Exception("rule_expression is not a RuleExpression object")
        validate_expr(rate_forward, "forward rate")
        if rule_expression.is_reversible:
            validate_expr(rate_reverse, "reverse rate")
        self.rule_expression = rule_expression
        self.reactant_pattern = rule_expression.reactant_pattern
        self.product_pattern = rule_expression.product_pattern
        self.is_reversible = rule_expression.is_reversible
        self.rate_forward = rate_forward
        self.rate_reverse = rate_reverse
        self.delete_molecules = delete_molecules
        self.move_connected = move_connected
        # TODO: ensure all numbered sites are referenced exactly twice within each of reactants and products

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



class Observable(Component, sympy.Symbol):

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
        return super(sympy.Symbol, cls).__new__(cls, name)

    def __getnewargs__(self):
        return (self.name, self.reaction_pattern, self.match, False)

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

    # This is needed to make sympy's evalf machinery treat this class like a
    # Symbol.
    @property
    def func(self):
        return sympy.Symbol

    def expand_obs(self):
        """ Expand observables in terms of species and coefficients """
        return sympy.Add(*[a * b for a, b in zip(
            self.coefficients,
            sympy.symbols(','.join('__s%d' % sp_id for sp_id in
                                   self.species) + ',')
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


class Expression(Component, sympy.Symbol):

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
        return super(sympy.Symbol, cls).__new__(cls, name)

    def __getnewargs__(self):
        return (self.name, self.expr, False)

    def __init__(self, name, expr, _export=True):
        Component.__init__(self, name, _export)
        self.expr = expr

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
        return self.expr.evalf()

    # This is needed to make sympy's evalf machinery treat this class like a
    # Symbol.
    @property
    def func(self):
        return sympy.Symbol

    def __repr__(self):
        ret = '%s(%s, %s)' % (self.__class__.__name__, repr(self.name),
                              repr(self.expr))
        return ret

    def __str__(self):
        return repr(self)


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
    initial_conditions : list of tuple of (ComplexPattern, Parameter)
        Specifies which species are present in the model's starting
        state (t=0) and how much there is of each one.  The
        ComplexPattern defines the species identity, and it must be
        concrete (see ComplexPattern.is_concrete).  The
        Parameter defines the amount or concentration of the species.
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
                        Expression)

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
        self.initial_conditions = []
        self.annotations = []
        self._odes = OdeView(self)
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
            self.initial_conditions = model_copy.initial_conditions

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

    def parameters_rules(self):
        """Return a ComponentSet of the parameters used in rules."""
        # rate_reverse is None for irreversible rules, so we'll need to filter those out
        cset = ComponentSet(p for r in self.rules for p in (r.rate_forward, r.rate_reverse)
                            if p is not None)
        # intersect with original parameter list to retain ordering
        return self.parameters & cset

    def parameters_initial_conditions(self):
        """Return a ComponentSet of initial condition parameters."""
        cset = ComponentSet(ic[1] for ic in self.initial_conditions)
        # intersect with original parameter list to retain ordering
        return self.parameters & cset

    def parameters_compartments(self):
        """Return a ComponentSet of compartment size parameters."""
        cset = ComponentSet(c.size for c in self.compartments)
        # intersect with original parameter list to retain ordering
        return self.parameters & cset

    def parameters_unused(self):
        """Return a ComponentSet of unused parameters."""
        cset_used = self.parameters_rules() | self.parameters_initial_conditions() | self.parameters_compartments()
        return self.parameters - cset_used

#     def expressions_constant(self):
#         """Return a ComponentSet of constant expressions."""
#         cset = ComponentSet(e for e in self.expressions
#                             if all(isinstance(a, (Parameter, sympy.Number))
#                                    for a in e.expand_expr().atoms()))
#         return cset
    
    def expressions_constant(self):
        """Return a ComponentSet of constant expressions."""
        cset = ComponentSet(e for e in self.expressions
                            if e.is_constant_expression())
        return cset

    def expressions_dynamic(self):
        """Return a ComponentSet of non-constant expressions."""
        return self.expressions - self.expressions_constant()

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

    def _validate_initial_condition_pattern(self, pattern):
        """
        Make sure a pattern is valid for an initial condition.

        Patterns must satisfy all of the following:
        * Able to be cast as a ComplexPattern
        * Concrete (see ComplexPattern.is_concrete)
        * Distinct from any existing initial condition pattern
        * match_once is False (nonsensical in this context)

        Parameters
        ----------
        pattern : MonomerPattern or ComplexPattern
            Pattern to validate

        Returns
        -------
        The validated pattern, upgraded to a ComplexPattern.

        """
        try:
            complex_pattern = as_complex_pattern(pattern)
        except InvalidComplexPatternException as e:
            raise InvalidInitialConditionError("Not a ComplexPattern")
        if not complex_pattern.is_concrete():
            raise InvalidInitialConditionError("Pattern not concrete")
        if any(complex_pattern.is_equivalent_to(other_cp)
               for other_cp, value in self.initial_conditions):
            # FIXME until we get proper canonicalization this could produce
            # false negatives
            raise InvalidInitialConditionError("Duplicate species")
        if complex_pattern.match_once:
            raise InvalidInitialConditionError("MatchOnce not allowed here")
        return complex_pattern

    def initial(self, pattern, value):
        """
        Add an initial condition.

        An initial condition is made up of a species and its amount or
        concentration.

        Parameters
        ----------
        pattern : ComplexPattern
            A concrete pattern defining the species to initialize.
        value : Parameter
            Amount of the species the model will start with.

        """
        complex_pattern = self._validate_initial_condition_pattern(pattern)
        validate_const_expr(value, "initial condition value")
        self.initial_conditions.append( (complex_pattern, value) )

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

        # Get the initial condition index
        ic_index_list = [i for i, ic in enumerate(self.initial_conditions)
                   if ic[0].is_equivalent_to(as_complex_pattern(before_pattern))]

        # If the initial condition to replace is not found, raise an error
        if not ic_index_list:
            raise ValueError("No initial condition found for pattern %s" %
                             before_pattern)

        # If more than one matching initial condition is found, raise an
        # error (this should never happen, because duplicate initial conditions
        # are not allowed to be created)
        assert len(ic_index_list) == 1
        ic_index = ic_index_list[0]

        # Make sure the new initial condition pattern is valid
        after_pattern = self._validate_initial_condition_pattern(after_pattern)

        # Since everything checks out, replace the old initial condition
        # pattern with the new one.  Because initial_conditions are tuples (and
        # hence immutable), we cannot simply replace the pattern; instead we
        # must delete the old one and add the new one.
        # We retain the old parameter object:
        p = self.initial_conditions[ic_index][1]
        del self.initial_conditions[ic_index]
        self.initial_conditions.append( (after_pattern, p) )

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


class ComponentSet(collections.Set, collections.Mapping, collections.Sequence):
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
        if isinstance(key, (int, long, slice)):
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

    def get(self, key, default=None):
        if isinstance(key, (int, long)):
            raise ValueError("get is undefined for integer arguments, use []"
                             "instead")
        try:
            return self[key]
        except KeyError:
            return default

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
            return collections.Set.__and__(self, other)
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


class OdeView(collections.Sequence):
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
