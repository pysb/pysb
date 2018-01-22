import collections
from .core import ComplexPattern, MonomerPattern, Monomer, \
    ReactionPattern, ANY, as_complex_pattern, DanglingBondError
import networkx as nx
from networkx.algorithms.isomorphism.vf2userfunc import GraphMatcher
from networkx.algorithms.isomorphism import categorical_node_match
import numpy as np
try:
    basestring
except NameError:
    # Under Python 3, do not pretend that bytes are a valid string
    basestring = str


def get_half_bonds_in_pattern(pat):
    """
    Return the list of integer bond numbers used in a pattern

    To return as a set, use :func:`get_bonds_in_pattern`.

    Parameters
    ----------
    pat : ComplexPattern, MonomerPattern, or None
        A pattern from which bond numberings are extracted

    Returns
    -------
    list
        Bond numbers used in the supplied pattern

    Examples
    --------

    >>> A = Monomer('A', ['b1', 'b2'], _export=False)
    >>> get_half_bonds_in_pattern(A(b1=None, b2=None))
    []
    >>> get_half_bonds_in_pattern(A(b1=1) % A(b2=1))
    [1, 1]
    """
    bonds_used = list()

    def _get_bonds_in_monomer_pattern(mp):
        for sc in mp.site_conditions.values():
            if isinstance(sc, int):
                bonds_used.append(sc)
            elif not isinstance(sc, basestring) and \
                    isinstance(sc, collections.Iterable):
                [bonds_used.append(b) for b in sc if isinstance(b, int)]

    if pat is None:
        return bonds_used
    if isinstance(pat, MonomerPattern):
        _get_bonds_in_monomer_pattern(pat)
    elif isinstance(pat, ComplexPattern):
        for mp in pat.monomer_patterns:
            _get_bonds_in_monomer_pattern(mp)
    else:
        raise ValueError('Unknown pattern type: %s' % type(pat))

    return bonds_used


def get_bonds_in_pattern(pat):
    """
    Return the set of integer bond numbers used in a pattern

    To return as a list (with duplicates), use
    :func:`get_half_bonds_in_pattern`

    Parameters
    ----------
    pat : ComplexPattern, MonomerPattern, or None
        A pattern from which bond numberings are extracted

    Returns
    -------
    set
        Bond numbers used in the supplied pattern

    Examples
    --------

    >>> A = Monomer('A', ['b1', 'b2'], _export=False)
    >>> get_bonds_in_pattern(A(b1=None, b2=None)) == set()
    True
    >>> get_bonds_in_pattern(A(b1=1) % A(b2=1)) == {1}
    True
    >>> get_bonds_in_pattern(A(b1=1) % A(b1=2, b2=1) % A(b1=2)) == {1, 2}
    True
    """
    return set(get_half_bonds_in_pattern(pat))


def check_dangling_bonds(pattern):
    """
    Check for dangling bonds in a PySB ComplexPattern/ReactionPattern

    Raises a DanglingBondError if a dangling bond is found
    """
    if isinstance(pattern, ReactionPattern):
        for cp in pattern.complex_patterns:
            check_dangling_bonds(cp)
        return
    bond_counts = collections.Counter(get_half_bonds_in_pattern(pattern))

    dangling_bonds = [bond for bond, count in bond_counts.items()
                      if count == 1]
    if dangling_bonds:
        raise DanglingBondError('Dangling bond(s) {} in {}'
                                .format(dangling_bonds, pattern))


def _match_graphs(pattern, candidate, exact):
    """ Compare two pattern graphs for isomorphism """
    node_matcher = categorical_node_match('id', default=None)
    if exact:
        return nx.is_isomorphic(pattern._as_graph(),
                                candidate._as_graph(),
                                node_match=node_matcher)
    else:
        return GraphMatcher(
            candidate._as_graph(), pattern._as_graph(),
            node_match=node_matcher
        ).subgraph_is_isomorphic()


def match_complex_pattern(pattern, candidate, exact=False):
    """
    Compare two ComplexPatterns against each other

    Parameters
    ----------
    pattern: pysb.ComplexPattern
    candidate: pysb.Complex.Pattern
    exact: bool
        Set to True for exact matches (i.e. species equivalence,
        or exact graph isomorphism). Set to False to compare as a
        pattern (i.e. subgraph isomorphism).

    Returns
    -------
    True if pattern matches candidate, False otherwise
    """
    if exact:
        if not pattern.is_concrete():
            raise ValueError('Pattern must be concrete for '
                             'exact matching: {}'.format(pattern))
        if not candidate.is_concrete():
            raise ValueError('Candidate must be concrete for '
                             'exact matching: {}'.format(candidate))

    if exact and len(pattern.monomer_patterns) != len(
            candidate.monomer_patterns):
        return False

    # Compare the monomer counts in the patterns so we can fail fast
    # without having to compare bonds using graph isomorphism checks, which
    # are more computationally expensive
    mons_pat = collections.Counter([mp.monomer for mp in \
            pattern.monomer_patterns])
    mons_cand = collections.Counter([mp.monomer for mp in \
            candidate.monomer_patterns])

    for mon, mon_count_cand in mons_cand.items():
        mon_count_pat = mons_pat.get(mon, 0)
        if exact and mon_count_cand != mon_count_pat:
            return False
        if mon_count_pat > mon_count_cand:
            return False

    # If we've got this far, we'll need to do a full pattern match
    # by searching for a graph isomorphism
    return _match_graphs(pattern, candidate, exact=exact)


def match_reaction_pattern(pattern, candidate):
    """
    Compare two ReactionPatterns against each other

    This function tests that every ComplexPattern in pattern has a
    matching ComplexPattern in candidate. If there's a one-to-one
    mapping of ComplexPattern matches, this is straightforward.
    Otherwise, we need to check for a maximum matching - a graph theory
    term referring to the maximum number of edges possible in a
    bipartite graph (representing ComplexPattern compatibility between
    pattern and candidate) without overlapping nodes. If every
    ComplexPattern in pattern has a match, then return True, otherwise
    return False. This algorithm is polynomial time (although the
    ComplexPattern isomorphism comparisons using match_complex_pattern are
    not).

    Parameters
    ----------
    pattern: pysb.ReactionPattern
    candidate: pysb.ReactionPattern

    Returns
    -------
    True if pattern matches candidate, False otherwise.

    """
    if len(pattern.complex_patterns) > len(candidate.complex_patterns):
        return False

    matches = []
    for cplx_pat in pattern.complex_patterns:
        matches_this = [cand_cplx_pat.matches(
            cplx_pat) for cand_cplx_pat in
            candidate.complex_patterns]
        matches_this = set(np.where(matches_this)[0])
        if len(matches_this) == 0:
            return False
        matches.append(matches_this)

    # If a unique 1:1 mapping exists, a match is assured
    if len(set.intersection(*matches)) == 0:
        return True

    # Find the maximum matching in a bipartite graph representing the
    # two sets of ComplexPatterns
    g = nx.Graph()
    g.add_nodes_from(['p%d' % n for n in
                     range(len(pattern.complex_patterns))], bipartite=0)
    g.add_nodes_from(['c%d' % n for n in
                     range(len(candidate.complex_patterns))], bipartite=1)
    for src_pat_id, src_pat_matches in enumerate(matches):
        g.add_edges_from([('p%d' % src_pat_id, 'c%d' % cand_pat_id) for
                          cand_pat_id in src_pat_matches])

    return (len(nx.bipartite.maximum_matching(g)) // 2) == len(
        pattern.complex_patterns)


def monomers_from_pattern(pattern):
    """ Return the set of monomers used in a pattern """
    if isinstance(pattern, ReactionPattern):
        return set.union(*[monomers_from_pattern(cp)
                           for cp in pattern.complex_patterns])
    if isinstance(pattern, ComplexPattern):
        return set([mp.monomer for mp in pattern.monomer_patterns])
    elif isinstance(pattern, MonomerPattern):
        return {pattern.monomer}
    elif isinstance(pattern, Monomer):
        return {pattern}
    else:
        raise Exception('Unsupported pattern type: %s' % type(pattern))


class SpeciesPatternMatcher(object):
    """
    Match a pattern against a model's species list

    Examples
    --------

    Create a PatternMatcher for the EARM 1.0 model

    >>> from pysb.examples.earm_1_0 import model
    >>> from pysb.bng import generate_equations
    >>> from pysb.pattern import SpeciesPatternMatcher
    >>> from pysb import ANY, WILD, Model, Monomer, as_complex_pattern
    >>> generate_equations(model)
    >>> spm = SpeciesPatternMatcher(model)

    Assign two monomers to variables (only needed when importing a model
    instead of defining one interactively)

    >>> Bax4 = model.monomers['Bax4']
    >>> Bcl2 = model.monomers['Bcl2']

    Search using a Monomer

    >>> spm.match(Bax4)
    [Bax4(b=None), Bax4(b=1) % Bcl2(b=1), Bax4(b=1) % Mito(b=1)]
    >>> spm.match(Bcl2) # doctest:+NORMALIZE_WHITESPACE
    [Bax2(b=1) % Bcl2(b=1),
    Bax4(b=1) % Bcl2(b=1),
    Bcl2(b=None),
    Bcl2(b=1) % MBax(b=1)]

    Search using a MonomerPattern (ANY and WILD keywords can be used)

    >>> spm.match(Bax4(b=WILD))
    [Bax4(b=None), Bax4(b=1) % Bcl2(b=1), Bax4(b=1) % Mito(b=1)]
    >>> spm.match(Bcl2(b=ANY))
    [Bax2(b=1) % Bcl2(b=1), Bax4(b=1) % Bcl2(b=1), Bcl2(b=1) % MBax(b=1)]

    Search using a ComplexPattern

    >>> spm.match(Bax4(b=1) % Bcl2(b=1))
    [Bax4(b=1) % Bcl2(b=1)]
    >>> spm.match(Bax4() % Bcl2())
    [Bax4(b=1) % Bcl2(b=1)]

    Contrived example to test a site with both a bond and state defined

    >>> model = Model(_export=False)
    >>> A = Monomer('A', ['a'], {'a': ['u', 'p']}, _export=False)
    >>> model.add_component(A)
    >>> species = [                                                     \
            A(a=None),                                                  \
            A(a='u'),                                                   \
            A(a=1) % A(a=1),                                            \
            A(a=('u', 1)) % A(a=('u', 1)),                              \
            A(a=('p', 1)) % A(a=('p', 1))                               \
        ]
    >>> model.species = [as_complex_pattern(sp) for sp in species]
    >>> spm2 = SpeciesPatternMatcher(model)
    >>> spm2.match(A()) # doctest:+NORMALIZE_WHITESPACE
    [A(a=None), A(a='u'), A(a=1) % A(a=1), A(a=('u', 1)) % A(a=('u', 1)),
     A(a=('p', 1)) % A(a=('p', 1))]
    >>> spm2.match(A(a='u'))
    [A(a='u')]
    >>> spm2.match(A(a=('u', ANY)))
    [A(a=('u', 1)) % A(a=('u', 1))]
    >>> spm2.match(A(a=('u', WILD)))
    [A(a='u'), A(a=('u', 1)) % A(a=('u', 1))]
    """
    def __init__(self, model, species=None):
        self.model = model
        if not species and not model.species:
            raise Exception('Model needs species list - run '
                            'generate_equations() first')

        if not species:
            species = model.species

        self.species = species

        self._species_cache = collections.defaultdict(set)
        for idx, sp in enumerate(species):
            self._add_species(idx, sp)

    def _add_species(self, idx, sp):
        if sp.compartment:
            raise NotImplementedError
        for mp in sp.monomer_patterns:
            if mp.compartment:
                raise NotImplementedError
            self._species_cache[mp.monomer].add(idx)

    def add_species(self, species, check_duplicate=True):
        """
        Add a species to the search list without adding to the model

        Parameters
        ----------
        species : ComplexPattern
            A concrete ComplexPattern (molecular species) to add to the
            search list
        check_duplicate : bool, optional
            If True, check the species list to make sure the new species
            is not already in the list
        """
        if check_duplicate and self.match(species, exact=True):
            return
        self.species.append(species)
        self._add_species(len(self.species) - 1, species)

    def match(self, pattern, index=False, exact=False):
        """
        Match a pattern against the list of species

        Parameters
        ----------
        pattern: pysb.Monomer or pysb.MonomerPattern or pysb.ComplexPattern
        index: bool
            If True, return species numerical index, rather than species itself
        exact: bool
            Treat Match as exact equivalence, not a pattern match (i.e. must be
            concrete if a MonomerPattern or ComplexPattern)

        Returns
        -------
        list of pysb.ComplexPattern or list of int
            A list of species matching the pattern is returned, unless
            index=True, in which case a list of the numerical indices of
            matching species is returned instead

        Examples
        --------

        >>> from pysb.examples import earm_1_0
        >>> from pysb.bng import generate_equations
        >>> model = earm_1_0.model
        >>> generate_equations(model)
        >>> spm = SpeciesPatternMatcher(model)
        >>> L = model.monomers['L']
        >>> spm.match(L())
        [L(b=None), L(b=1) % pR(b=1)]
        """
        if not isinstance(pattern, (Monomer, MonomerPattern, ComplexPattern)):
            raise ValueError('A Monomer, MonomerPattern or ComplexPattern is '
                             'required to match species')

        monomers = monomers_from_pattern(pattern)

        if exact:
            if isinstance(pattern, (Monomer, MonomerPattern)):
                num_mon_pats = 1
            else:
                num_mon_pats = len(pattern.monomer_patterns)
        else:
            # Don't check the number of monomer patterns in search
            # candidates if we're not doing an exact match of the species
            num_mon_pats = None

        shortlist, shortlist_indexes = self._species_containing_monomers(
            monomers, num_mon_pats)

        # If pattern is a Monomer, we're done
        if isinstance(pattern, Monomer):
            return shortlist_indexes if index else shortlist
        else:
            return [(shortlist_indexes[idx] if index else sp) for idx, sp in
                    enumerate(shortlist) if
                    match_complex_pattern(
                        as_complex_pattern(pattern), sp, exact=exact
                    )]

    def _species_containing_monomers(self, monomer_list, num_mon_pats=None):
        """
        Identifies species containing a list of monomers

        Parameters
        ----------
        monomer_list: list of Monomers
            A list of monomers with which to search the model's species
        num_mon_pats: int or None
            Restrict matches to species with exactly the specified number of
            MonomerPatterns

        Returns
        -------
        Model species containing all of the monomers in the list
        """
        sp_indexes = set.intersection(*[self._species_cache[mon] for mon in
                                        monomer_list])
        if num_mon_pats:
            retval = zip(*[(self.species[sp], sp) for sp in sp_indexes
                           if len(self.species[sp].monomer_patterns)
                           == num_mon_pats])
            return retval if retval else ((), ())
        else:
            return [self.species[sp] for sp in sp_indexes], list(sp_indexes)

    def rule_firing_species(self, rules_to_consider=None,
                            include_reverse=True):
        """
        Return the species which match the reactants of a set of rules

        Parameters
        ----------
        rules_to_consider: list of pysb.Rule or None
            A list of rules to use. If None, use all rules in the model.
        include_reverse: bool, optional
            For reversible rules, include species triggering the rule in
            reverse

        Returns
        -------
        collections.OrderedDict
            Dictionary of PySB rules whose reactants contain at least one of
            the species in the model. Keys are PySB rules, values are a list
            of lists. Each outer list corresponding to each
            ComplexPattern in the ReactantPattern (or ReactantPattern and
            ProductPattern, if rule is reversible). Each inner list contains
            the list of species matching the corresponding ComplexPattern.

        Examples
        --------

        >>> from pysb.examples import robertson
        >>> from pysb.bng import generate_equations
        >>> model = robertson.model
        >>> generate_equations(model)
        >>> spm = SpeciesPatternMatcher(model)

        Get a list of species which fire each rule:

        >>> spm.rule_firing_species() \
                #doctest: +NORMALIZE_WHITESPACE
        OrderedDict([(Rule('A_to_B', A() >> B(), k1), [[A()]]),
         (Rule('BB_to_BC', B() + B() >> B() + C(), k2), [[B()], [B()]]),
         (Rule('BC_to_AC', B() + C() >> A() + C(), k3), [[B()], [C()]])])
        """
        if rules_to_consider is None:
            rules_to_consider = self.model.rules
        rules_fired = collections.OrderedDict()
        for r in rules_to_consider:
            rp = r.reactant_pattern
            if len(rp.complex_patterns) == 0:
                # Synthesis rules are always fired
                rules_fired[r] = []
            else:
                species_fired = self.species_fired_by_reactant_pattern(rp)
                if include_reverse and r.is_reversible:
                    species_fired += self.species_fired_by_reactant_pattern(
                        r.product_pattern)
                if species_fired:
                    rules_fired[r] = species_fired
        return rules_fired

    def species_fired_by_reactant_pattern(self, reaction_pattern):
        """
        Get list of species matching a reactant pattern

        Parameters
        ----------
        reaction_pattern: pysb.ReactionPattern

        Returns
        -------
        list of lists of pysb.ComplexPattern
            List of lists of species matching each ComplexPattern in the
            ReactantPattern.

        Examples
        --------

        >>> from pysb.examples import bax_pore
        >>> from pysb.bng import generate_equations
        >>> model = bax_pore.model
        >>> generate_equations(model)
        >>> spm = SpeciesPatternMatcher(model)

        Get a list of species which fire each rule:

        >>> rxn_pat = model.rules['bax_dim'].reactant_pattern
        >>> print(rxn_pat)
        BAX(t1=None, t2=None) + BAX(t1=None, t2=None)

        >>> spm.species_fired_by_reactant_pattern(rxn_pat) \
                #doctest: +NORMALIZE_WHITESPACE
        [[BAX(t1=None, t2=None, inh=None),
          BAX(t1=None, t2=None, inh=1) % MCL1(b=1)],
         [BAX(t1=None, t2=None, inh=None),
              BAX(t1=None, t2=None, inh=1) % MCL1(b=1)]]
        """
        species_fired = []

        for i, cp in enumerate(reaction_pattern.complex_patterns):
            species_fired_this_cp = self.match(cp)
            if not species_fired_this_cp:
                return []
            else:
                species_fired.append(species_fired_this_cp)

        return species_fired


class RulePatternMatcher(object):
    """
    Match a pattern against a model's species list

    Methods are provided to match against rule reactants, products or both.
    Searches can be Monomers, MonomerPatterns, ComplexPatterns or
    ReactionPatterns.

    Examples
    --------

    Create a PatternMatcher for the EARM 1.0 model

    >>> from pysb.examples.earm_1_0 import model
    >>> from pysb.pattern import RulePatternMatcher
    >>> rpm = RulePatternMatcher(model)

    Assign some monomers to variables (only needed when importing a model
    instead of defining one interactively)

    >>> AMito, mCytoC, mSmac, cSmac = [model.monomers[m] for m in \
        ('AMito', 'mCytoC', 'mSmac', 'cSmac')]

    Search using a Monomer

    >>> rpm.match_reactants(AMito) # doctest:+NORMALIZE_WHITESPACE
    [Rule('bind_mCytoC_AMito', AMito(b=None) + mCytoC(b=None) |
        AMito(b=1) % mCytoC(b=1), kf20, kr20),
    Rule('produce_ACytoC_via_AMito', AMito(b=1) % mCytoC(b=1) >>
        AMito(b=None) + ACytoC(b=None), kc20),
    Rule('bind_mSmac_AMito', AMito(b=None) + mSmac(b=None) |
        AMito(b=1) % mSmac(b=1), kf21, kr21),
    Rule('produce_ASmac_via_AMito', AMito(b=1) % mSmac(b=1) >>
        AMito(b=None) + ASmac(b=None), kc21)]

    >>> rpm.match_products(mSmac) # doctest:+NORMALIZE_WHITESPACE
    [Rule('bind_mSmac_AMito', AMito(b=None) + mSmac(b=None) |
        AMito(b=1) % mSmac(b=1), kf21, kr21)]

    Search using a MonomerPattern

    >>> rpm.match_reactants(AMito(b=1)) # doctest:+NORMALIZE_WHITESPACE
    [Rule('produce_ACytoC_via_AMito', AMito(b=1) % mCytoC(b=1) >>
        AMito(b=None) + ACytoC(b=None), kc20),
    Rule('produce_ASmac_via_AMito', AMito(b=1) % mSmac(b=1) >>
        AMito(b=None) + ASmac(b=None), kc21)]

    >>> rpm.match_rules(cSmac(b=1)) # doctest:+NORMALIZE_WHITESPACE
    [Rule('inhibit_cSmac_by_XIAP', cSmac(b=None) + XIAP(b=None) |
        cSmac(b=1) % XIAP(b=1), kf28, kr28)]

    Search using a ComplexPattern

    >>> rpm.match_reactants(AMito() % mSmac()) # doctest:+NORMALIZE_WHITESPACE
    [Rule('produce_ASmac_via_AMito', AMito(b=1) % mSmac(b=1) >>
        AMito(b=None) + ASmac(b=None), kc21)]

    >>> rpm.match_rules(AMito(b=1) % mCytoC(b=1)) \
        # doctest:+NORMALIZE_WHITESPACE
    [Rule('bind_mCytoC_AMito', AMito(b=None) + mCytoC(b=None) |
        AMito(b=1) % mCytoC(b=1), kf20, kr20),
    Rule('produce_ACytoC_via_AMito', AMito(b=1) % mCytoC(b=1) >>
        AMito(b=None) + ACytoC(b=None), kc20)]

    Search using a ReactionPattern

    >>> rpm.match_reactants(mCytoC() + mSmac())
    []

    >>> rpm.match_reactants(AMito() + mCytoC()) # doctest:+NORMALIZE_WHITESPACE
    [Rule('bind_mCytoC_AMito', AMito(b=None) + mCytoC(b=None) |
        AMito(b=1) % mCytoC(b=1), kf20, kr20)]

    """

    def __init__(self, model):
        self.model = model

        self._reactant_cache = collections.defaultdict(set)
        self._product_cache = collections.defaultdict(set)

        for rule in model.rules:
            for cache, rp in ((self._reactant_cache, rule.reactant_pattern),
                              (self._product_cache, rule.product_pattern)):
                for cp in rp.complex_patterns:
                    if cp.compartment:
                        raise NotImplementedError
                    for mp in cp.monomer_patterns:
                        if mp.compartment:
                            raise NotImplementedError
                        cache[mp.monomer].add(rule.name)

    def match_reactants(self, pattern):
        return self._match_reaction_patterns(pattern, 'reactant')

    def match_products(self, pattern):
        return self._match_reaction_patterns(pattern, 'product')

    def match_rules(self, pattern):
        return [r for r in self.model.rules if
                r in self.match_reactants(pattern) or
                r in self.match_products(pattern)]

    def _match_reaction_patterns(self, pattern, reaction_side):
        if not isinstance(pattern, (Monomer, MonomerPattern, ComplexPattern,
                                    ReactionPattern)):
            raise ValueError('A Monomer, MonomerPattern, ComplexPattern or '
                             'ReactionPattern required to match rules')

        monomers = monomers_from_pattern(pattern)

        if reaction_side == 'reactant':
            cache = self._reactant_cache

            def pat_fn(r):
                return r.reactant_pattern
        elif reaction_side == 'product':
            cache = self._product_cache

            def pat_fn(r):
                return r.product_pattern
        else:
            raise Exception('reaction_side must be "reactant" or "product"')

        shortlist = self._cache_containing_monomers(cache, monomers)

        # If pattern is a Monomer, we're done
        if isinstance(pattern, Monomer):
            return shortlist

        if isinstance(pattern, (MonomerPattern, ComplexPattern)):
            new_shortlist = []
            for rule in shortlist:
                reaction_pattern = pat_fn(rule)
                if self._match_complex_pattern_to_reaction_pattern(
                        as_complex_pattern(pattern), reaction_pattern):
                    new_shortlist.append(rule)

            return new_shortlist

        else:
            return [rule for rule in shortlist if
                    pat_fn(rule).matches(pattern)]

    @classmethod
    def _match_complex_pattern_to_reaction_pattern(cls, pattern, test_pattern):
        for cp in test_pattern.complex_patterns:
            if match_complex_pattern(pattern, cp):
                return True
        return False

    def _cache_containing_monomers(self, cache, monomer_list):
        """
        Identifies rules containing a list of monomers

        Parameters
        ----------
        monomer_list: list of Monomers
            A list of monomers with which to search the model's rules

        Returns
        -------
        Model rules containing all of the monomers in the list

        """
        rule_names = set.intersection(*[cache[mon] for mon in
                                        monomer_list])
        return [r for r in self.model.rules if r.name in rule_names]


class ReactionPatternMatcher(object):
    """
    Match a pattern against a model's reactions list

    Methods are provided to match against reaction reactants, products or
    both. Searches can be Monomers, MonomerPatterns, ComplexPatterns or
    ReactionPatterns.

    Examples
    --------

    Create a PatternMatcher for the EARM 1.0 model

    >>> from pysb.examples.earm_1_0 import model
    >>> from pysb.bng import generate_equations
    >>> from pysb.pattern import ReactionPatternMatcher
    >>> generate_equations(model)
    >>> rpm = ReactionPatternMatcher(model)

    Assign some monomers to variables (only needed when importing a model
    instead of defining one interactively)

    >>> AMito, mCytoC, mSmac, cSmac = [model.monomers[m] for m in \
                                       ('AMito', 'mCytoC', 'mSmac', 'cSmac')]

    Search using a Monomer

    >>> rpm.match_products(mSmac) # doctest:+NORMALIZE_WHITESPACE
    [Rxn (reversible):
        Reactants: {'__s15': mSmac(b=None), '__s45': AMito(b=None)}
        Products: {'__s47': AMito(b=1) % mSmac(b=1)}
        Rate: __s15*__s45*kf21 - __s47*kr21
        Rules: [Rule('bind_mSmac_AMito', AMito(b=None) + mSmac(b=None) |
                AMito(b=1) % mSmac(b=1), kf21, kr21)]]

    Search using a MonomerPattern

    >>> rpm.match_reactants(AMito(b=ANY)) # doctest:+NORMALIZE_WHITESPACE
    [Rxn (one-way):
        Reactants: {'__s46': AMito(b=1) % mCytoC(b=1)}
        Products: {'__s45': AMito(b=None), '__s48': ACytoC(b=None)}
        Rate: __s46*kc20
        Rules: [Rule('produce_ACytoC_via_AMito', AMito(b=1) % mCytoC(b=1) >>
                AMito(b=None) + ACytoC(b=None), kc20)],
     Rxn (one-way):
        Reactants: {'__s47': AMito(b=1) % mSmac(b=1)}
        Products: {'__s45': AMito(b=None), '__s49': ASmac(b=None)}
        Rate: __s47*kc21
        Rules: [Rule('produce_ASmac_via_AMito', AMito(b=1) % mSmac(b=1) >>
                AMito(b=None) + ASmac(b=None), kc21)]]

    >>> rpm.match_products(cSmac(b=ANY)) # doctest:+NORMALIZE_WHITESPACE
    [Rxn (reversible):
        Reactants: {'__s7': XIAP(b=None), '__s51': cSmac(b=None)}
        Products: {'__s53': XIAP(b=1) % cSmac(b=1)}
        Rate: __s51*__s7*kf28 - __s53*kr28
        Rules: [Rule('inhibit_cSmac_by_XIAP', cSmac(b=None) + XIAP(b=None) |
                cSmac(b=1) % XIAP(b=1), kf28, kr28)]]

    Search using a ComplexPattern

    >>> rpm.match_reactants(AMito() % mSmac()) # doctest:+NORMALIZE_WHITESPACE
    [Rxn (one-way):
        Reactants: {'__s47': AMito(b=1) % mSmac(b=1)}
        Products: {'__s45': AMito(b=None), '__s49': ASmac(b=None)}
        Rate: __s47*kc21
        Rules: [Rule('produce_ASmac_via_AMito', AMito(b=1) % mSmac(b=1) >>
                AMito(b=None) + ASmac(b=None), kc21)]]

    >>> rpm.match_reactions(AMito(b=3) % mCytoC(b=3)) \
    # doctest:+NORMALIZE_WHITESPACE
    [Rxn (reversible):
        Reactants: {'__s14': mCytoC(b=None), '__s45': AMito(b=None)}
        Products: {'__s46': AMito(b=1) % mCytoC(b=1)}
        Rate: __s14*__s45*kf20 - __s46*kr20
        Rules: [Rule('bind_mCytoC_AMito', AMito(b=None) + mCytoC(b=None) |
                AMito(b=1) % mCytoC(b=1), kf20, kr20)],
     Rxn (one-way):
        Reactants: {'__s46': AMito(b=1) % mCytoC(b=1)}
        Products: {'__s45': AMito(b=None), '__s48': ACytoC(b=None)}
        Rate: __s46*kc20
        Rules: [Rule('produce_ACytoC_via_AMito', AMito(b=1) % mCytoC(b=1) >>
                AMito(b=None) + ACytoC(b=None), kc20)]]
    """
    def __init__(self, model, species_pattern_matcher=None):
        self.model = model

        # In this cache, our caches map species to reactions
        self._reactant_cache = collections.defaultdict(set)
        self._product_cache = collections.defaultdict(set)

        if not species_pattern_matcher:
            self.spm = SpeciesPatternMatcher(model)

        for r_id, rxn in enumerate(model.reactions_bidirectional):
            for cache, species_ids in (
                    (self._reactant_cache, rxn['reactants']),
                    (self._product_cache, rxn['products'])):
                for sp_id in species_ids:
                    sp = model.species[sp_id]
                    if sp.compartment:
                        raise NotImplementedError
                    cache[sp].add(r_id)

    def match_reactants(self, pattern):
        return self._match_reactions_against_cache(pattern, 'reactant')

    def match_products(self, pattern):
        return self._match_reactions_against_cache(pattern, 'product')

    def match_reactions(self, pattern):
        return self._match_reactions_against_cache(pattern, 'both')

    def _match_reactions_against_cache(self, pattern, reaction_side):
        species = self.spm.match(pattern)

        rxn_ids = set()
        if reaction_side in ['reactant', 'both']:
            rxn_ids.update(*[self._reactant_cache[sp] for sp in species])

        if reaction_side in ['product', 'both']:
            rxn_ids.update(*[self._product_cache[sp] for sp in species])
        rxn_ids = list(rxn_ids)
        rxn_ids.sort()

        return [_Reaction(rxn_dict=self.model.reactions_bidirectional[rxn_id],
                         model=self.model) for rxn_id in rxn_ids]


class _Reaction(object):
    __slots__ = ['_rxn_dict', 'reactants', 'model', 'products', 'species']
    """
    Store reactions in object form for pretty-printing
    """
    def __init__(self, rxn_dict=None, model=None, species=None):
        self._rxn_dict = rxn_dict

        if model is None:
            raise ValueError('Must specify model or species list')

        self.model = model

        if species:
            self.species = species
        else:
            self.species = model.species

        self.reactants = collections.defaultdict(int)
        self.products = collections.defaultdict(int)

        for r_id in rxn_dict['reactants']:
            self.reactants[r_id] += 1

        for p_id in rxn_dict['products']:
            self.products[p_id] += 1

    @property
    def reversible(self):
        return self._rxn_dict.get('reversible', None)

    @property
    def reverse(self):
        return self._rxn_dict.get('reverse', None)

    @property
    def rules(self):
        return [self.model.rules[r] for r in self._rxn_dict['rule']]

    def add_rule(self, rule):
        if rule.name not in self._rxn_dict['rule']:
            self._rxn_dict['rule'].append(rule.name)

    @property
    def rate(self):
        return self._rxn_dict['rate']

    def __repr__(self):
        return 'Rxn (%s): \n    Reactants: %s\n    Products: %s\n    ' \
               'Rate: %s\n    Rules: %s' % (
                    'reversible' if self.reversible else
                    ('one-way [reverse]' if self.reverse else
                        'one-way'),
                    self._repr_species_dict(self.reactants),
                    self._repr_species_dict(self.products),
                    self.rate,
                    self.rules
               )

    def __cmp__(self, other):
        try:
            return self._rxn_dict == other._rxn_dict
        except AttributeError:
            return False

    def _repr_species_dict(self, species_dict):
        return '{%s}' % ', '.join(["'__s%d': %s" % (k, self.species[k])
                                   for k in sorted(species_dict.keys())])
