import inspect
from pysb import *
import pysb.core
from pysb.core import ComponentSet, as_reaction_pattern, as_complex_pattern
import numbers
import functools
import itertools

__all__ = ['equilibrate',
           'bind', 'bind_table',
           'catalyze', 'catalyze_state',
           'catalyze_one_step', 'catalyze_one_step_reversible',
           'synthesize', 'degrade', 'synthesize_degrade_table',
           'assemble_pore_sequential', 'pore_transport', 'pore_bind']

# Internal helper functions
# =========================

def _complex_pattern_label(cp):
    """Return a string label for a ComplexPattern."""
    mp_labels = [_monomer_pattern_label(mp) for mp in cp.monomer_patterns]
    return ''.join(mp_labels)

def _monomer_pattern_label(mp):
    """Return a string label for a MonomerPattern."""
    site_values = [str(x) for x in mp.site_conditions.values()
                            if x is not None
                            and not isinstance(x, list)
                            and not isinstance(x, tuple)
                            and not isinstance(x, numbers.Real)]
    return mp.monomer.name + ''.join(site_values)

def _rule_name_generic(rule_expression):
    """Return a generic string label for a RuleExpression."""
    # Get ReactionPatterns
    react_p = rule_expression.reactant_pattern
    prod_p = rule_expression.product_pattern
    # Build the label components
    lhs_label = [_complex_pattern_label(cp) for cp in react_p.complex_patterns]
    lhs_label = '_'.join(lhs_label)
    rhs_label = [_complex_pattern_label(cp) for cp in prod_p.complex_patterns]
    rhs_label = '_'.join(rhs_label)
    return '%s_to_%s' % (lhs_label, rhs_label)

def _macro_rule(rule_prefix, rule_expression, klist, ksuffixes,
                name_func=_rule_name_generic):
    """A helper function for writing macros that generates a single rule.

    Parameters
    ----------
    rule_prefix : string
        The prefix that is prepended to the (automatically generated) name for
        the rule.
    rule_expression : RuleExpression
        An expression specifying the form of the rule; gets passed directly
        to the Rule constructor.
    klist : list of Parameters or list of numbers
        If the rule is unidirectional, the list must contain one element
        (either a Parameter or number); if the rule is reversible, it must
        contain two elements. If the rule is reversible, the first element
        in the list is taken to be the forward rate, and the second element
        is taken as the reverse rate. 
    ksuffixes : list of strings
        If klist contains numbers rather than Parameters, the strings in
        ksuffixes are used to automatically generate the necessary Parameter
        objects. The suffixes are appended to the rule name to generate the
        associated parameter name. ksuffixes must contain one element if the
        rule is unidirectional, two if it is reversible.
    name_func : function, optional
        A function which takes a RuleExpression and returns a string label for
        it, to be called as part of the automatic rule name generation. If not
        provided, a built-in default naming function will be used.

    Returns
    -------
    components : ComponentSet
        The generated components. Contains the generated Rule and up to two
        generated Parameter objects (if klist was given as numbers).

    Notes
    -----
    The default naming scheme (if `name_func` is not passed) follows the form::

        '%s_%s_to_%s' % (rule_prefix, lhs_label, rhs_label)

    where lhs_label and rhs_label are each concatenations of the Monomer names
    and specified sites in the ComplexPatterns on each side of the
    RuleExpression. The actual implementation is in the function
    _rule_name_generic, which in turn calls _complex_pattern_label and
    _monomer_pattern_label. For some specialized reactions it may be helpful to
    devise a custom naming scheme rather than rely on this default.

    Examples
    --------
    Using distinct Monomers for substrate and product::

        >>> from pysb import *
        >>> from pysb.macros import _macro_rule
        >>> 
        >>> Model() # doctest:+ELLIPSIS
        <Model '<interactive>' (monomers: 0, rules: 0, parameters: 0, compartments: 0) at ...>
        >>> Monomer('A', ['s'])
        Monomer(name='A', sites=['s'], site_states={})
        >>> Monomer('B', ['s'])
        Monomer(name='B', sites=['s'], site_states={})
        >>> 
        >>> _macro_rule('bind', A(s=None) + B(s=None) <> A(s=1) % B(s=1), [1e6, 1e-1], ['kf', 'kr'])
        {'bind_A_B_to_A1B1': Rule(name='bind_A_B_to_A1B1', reactants=A(s=None) + B(s=None), products=A(s=1) % B(s=1), rate_forward=Parameter(name='bind_A_B_to_A1B1_kf', value=1000000.0), rate_reverse=Parameter(name='bind_A_B_to_A1B1_kr', value=0.1)),
         'bind_A_B_to_A1B1_kf': Parameter(name='bind_A_B_to_A1B1_kf', value=1000000.0),
         'bind_A_B_to_A1B1_kr': Parameter(name='bind_A_B_to_A1B1_kr', value=0.1)}
        >>> 

    """

    r_name = '%s_%s' % (rule_prefix, name_func(rule_expression))

    # If rule is unidirectional, make sure we only have one parameter
    if (not rule_expression.is_reversible):
        if len(klist) != 1 or len(ksuffixes) != 1:
            raise ValueError("A unidirectional rule must have one parameter.")
    # If rule is bidirectional, make sure we have two parameters
    else:
        if len(klist) != 2 or len(ksuffixes) != 2:
            raise ValueError("A bidirectional rule must have two parameters.")

    if all(isinstance(x, Parameter) for x in klist):
        k1 = klist[0]
        if rule_expression.is_reversible:
            k2 = klist[1]
        params_created = ComponentSet()
    # if klist is numbers, generate the Parameters
    elif all(isinstance(x, numbers.Real) for x in klist):
        k1 = Parameter('%s_%s' % (r_name, ksuffixes[0]), klist[0])
        params_created = ComponentSet([k1]) 
        if rule_expression.is_reversible:
            k2 = Parameter('%s_%s' % (r_name, ksuffixes[1]),
                           klist[1])
            params_created.add(k2)
    else:
        raise ValueError("klist must contain Parameter objects or numbers.")

    if rule_expression.is_reversible:
        r = Rule(r_name, rule_expression, k1, k2)
    else:
        r = Rule(r_name, rule_expression, k1)

    # Build a set of components that were created
    return ComponentSet([r]) | params_created

def _verify_sites(m, *site_list):
    """Checks that the monomer m contains all of the sites in site_list.

    Parameters
    ----------
    m : Monomer or MonomerPattern
        The monomer to check.
    site1, site2, ... : string
        One or more site names to check on m

    Returns
    -------
    True if m contains all sites; raises a ValueError otherwise.

    Raises
    ------
    ValueError
        If any of the sites are not found.
    """

    for site in site_list:
        if site not in m().monomer.sites_dict:
            raise ValueError("Monomer '%s' must contain the site '%s'" %
                            (m().monomer.name, site))
    return True

# Unimolecular patterns
# =====================

def equilibrate(s1, s2, klist):
    """Generate the unimolecular reversible equilibrium reaction S1 <-> S2.

    Parameters
    ----------
    s1, s2 : Monomer or MonomerPattern
        S1 and S2 in the above reaction.
    klist : list of 2 Parameters or list of 2 numbers
        Forward (S1 -> S2) and reverse rate constants (in that order). If
        Parameters are passed, they will be used directly in the generated
        Rules. If numbers are passed, Parameters will be created with
        automatically generated names based on the names and states of S1 and S2
        and these parameters will be included at the end of the returned
        component list.

    Returns
    -------
    components : ComponentSet
        The generated components. Contains one reversible Rule and optionally
        two Parameters if klist was given as plain numbers.
    """
    
    # turn any Monomers into MonomerPatterns
    return _macro_rule('equilibrate', s1 <> s2, klist, ['kf', 'kr'])

# Binding
# =======

def bind(s1, site1, s2, site2, klist):
    """Generate the reversible binding reaction S1 + S2 <> S1:S2.

    Parameters
    ----------
    s1, s2 : Monomer or MonomerPattern
        Monomers participating in the binding reaction.
    site1, site2 : string 
        The names of the sites on s1 and s2 used for binding.
    klist : list of 2 Parameters or list of 2 numbers
        Forward and reverse rate constants (in that order). If Parameters are
        passed, they will be used directly in the generated Rules. If numbers
        are passed, Parameters will be created with automatically generated
        names based on the names and states of S1 and S2 and these parameters
        will be included at the end of the returned component list.

    Returns
    -------
    components : ComponentSet
        The generated components. Contains the bidirectional binding Rule
        and optionally two Parameters if klist was given as numbers.

    Examples
    --------
        Model()
        Monomer('A', ['x'])
        Monomer('B', ['y'])
        bind(A, 'x', B, 'y', [1e-4, 1e-1])

    """

    _verify_sites(s1, site1)
    _verify_sites(s2, site2)

    def bind_name_func(rule_expression):
        # Get ComplexPatterns
        react_cps = rule_expression.reactant_pattern.complex_patterns
        # Build the label components
        return '_'.join(_complex_pattern_label(cp) for cp in react_cps)

    return _macro_rule('bind',
                       s1({site1: None}) + s2({site2: None}) <>
                       s1({site1: 1}) % s2({site2: 1}),
                       klist, ['kf', 'kr'], name_func=bind_name_func)

def bind_table(bindtable, row_site, col_site, kf=None):
    """Generate a table of reversible binding reactions.

    Given two lists of species R and C, calls the `bind` macro on each pairwise
    combination (R[i], C[j]). The species lists and the parameter values are
    passed as a list of lists (i.e. a table) with elements of R passed as the
    "row headers", elements of C as the "column headers", and forward / reverse
    rate pairs (in that order) as tuples in the "cells". For example with two
    elements in each of R and C, the table would appear as follows (note that
    the first row has one fewer element than the subsequent rows)::

        [[              C1,           C2],
         [R1, (1e-4, 1e-1), (2e-4, 2e-1)],
         [R2, (3e-4, 3e-1), (4e-4, 4e-1)]]

    Each parameter tuple may contain Parameters or numbers. If Parameters are
    passed, they will be used directly in the generated Rules. If numbers are
    passed, Parameters will be created with automatically generated names based
    on the names and states of the relevant species and these parameters will be
    included at the end of the returned component list. To omit any individual
    reaction, pass None in place of the corresponding parameter tuple.

    Alternately, single kd values (dissociation constant, kr/kf) may be
    specified instead of (kf, kr) tuples. If kds are used, a single shared kf
    Parameter or number must be passed as an extra `kf` argument. kr values for
    each binding reaction will be calculated as kd*kf. It is important to
    remember that the forward rate constant is a single parameter shared across
    the entire bind table, as this may have implications for parameter fitting.

    Parameters
    ----------
    bindtable : list of lists
        Table of reactants and rates, as described above.
    row_site, col_site : string 
        The names of the sites on the elements of R and C, respectively, used
        for binding.
    kf : Parameter or number, optional
        If the "cells" in bindtable are given as single kd values, this is the
        shared kf used to calculate the kr values.

    Returns
    -------
    components : ComponentSet
        The generated components. Contains the bidirectional binding Rules and
        optionally the Parameters for any parameters given as numbers.

    Examples
    --------
        Model()
        Monomer('R1', ['x'])
        Monomer('R2', ['x'])
        Monomer('C1', ['y'])
        Monomer('C2', ['y'])
        bind_table([[              C1,           C2],
                    [R1, (1e-4, 1e-1), (2e-4, 2e-1)],
                    [R2, (3e-4, 3e-1), None        ]],
                   'x', 'y')

    """

    # extract species lists and matrix of rates
    s_rows = [row[0] for row in bindtable[1:]]
    s_cols = bindtable[0]
    kmatrix = [row[1:] for row in bindtable[1:]]

    # ensure kf is passed when necessary
    kiter = itertools.chain.from_iterable(kmatrix)
    if any(isinstance(x, numbers.Real) for x in kiter) and kf is None:
        raise ValueError("must specify kf when using single kd values")

    # loop over interactions
    components = ComponentSet()
    for r, s_row in enumerate(s_rows):
        for c, s_col in enumerate(s_cols):
            klist = kmatrix[r][c]
            if klist is not None:
                # if user gave a single kd, calculate kr
                if isinstance(klist, numbers.Real):
                    kd = klist
                    klist = (kf, kd*kf)
                components |= bind(s_row(), row_site, s_col(), col_site, klist)

    return components

# Catalysis
# =========

def catalyze(enzyme, e_site, substrate, s_site, product, klist):
    """Generate the two-step catalytic reaction E + S <> E:S >> E + P.

    Parameters
    ----------
    enzyme, substrate, product : Monomer or MonomerPattern
        E, S and P in the above reaction.
    e_site, s_site : string
        The names of the sites on `enzyme` and `substrate` (respectively) where
        they bind each other to form the E:S complex.
    klist : list of 3 Parameters or list of 3 numbers
        Forward, reverse and catalytic rate constants (in that order). If
        Parameters are passed, they will be used directly in the generated
        Rules. If numbers are passed, Parameters will be created with
        automatically generated names based on the names and states of enzyme,
        substrate and product and these parameters will be included at the end
        of the returned component list.

    Returns
    -------
    components : ComponentSet
        The generated components. Contains two Rules (bidirectional complex
        formation and unidirectional product dissociation), and optionally three
        Parameters if klist was given as plain numbers.

    Notes
    -----
    When passing a MonomerPattern for `enzyme` or `substrate`, do not include
    `e_site` or `s_site` in the respective patterns. The macro will handle this.

    Examples
    --------
    Using distinct Monomers for substrate and product::

        Model()
        Monomer('E', ['b'])
        Monomer('S', ['b'])
        Monomer('P')
        catalyze(E, 'b', S, 'b', P, (1e-4, 1e-1, 1))

    Using a single Monomer for substrate and product with a state change::

        Monomer('Kinase', ['b'])
        Monomer('Substrate', ['b', 'y'], {'y': ('U', 'P')})
        catalyze(Kinase, 'b', Substrate(y='U'), 'b', Substrate(y='P'),
                 (1e-4, 1e-1, 1))
    """

    _verify_sites(enzyme, e_site)
    _verify_sites(substrate, s_site)

    # Set up some aliases to the patterns we'll use in the rules
    enzyme_free = enzyme({e_site: None})
    substrate_free = substrate({s_site: None})
    es_complex = enzyme({e_site: 1}) % substrate({s_site: 1})

    # if product is actually a variant of substrate, we need to explicitly say
    # that it is no longer bound to enzyme
    if product().monomer is substrate().monomer:
        product = product({e_site: None})

    # create the rules
    components = _macro_rule('bind',
                             enzyme_free + substrate_free <> es_complex,
                             klist[0:2], ['kf', 'kr'])
    components |= _macro_rule('catalyze',
                              es_complex >> enzyme_free + product,
                              [klist[2]], ['kc'])

    return components

def catalyze_state(enzyme, e_site, substrate, s_site, mod_site,
                   state1, state2, klist):
    """Generate the two-step catalytic reaction E + S <> E:S >> E + P.
    A wrapper around catalyze() with a signature specifying the state change
    of the substrate resulting from catalysis.

    Parameters
    ----------
    enzyme : Monomer or MonomerPattern
        E in the above reaction.
    substrate : Monomer or MonomerPattern
        S and P in the above reaction. The product species is assumed to be
        identical to the substrate species in all respects except the state
        of the modification site. The state of the modification site should
        not be specified in the MonomerPattern for the substrate.
    e_site, s_site : string
        The names of the sites on `enzyme` and `substrate` (respectively) where
        they bind each other to form the E:S complex.
    mod_site : string
        The name of the site on the substrate that is modified by catalysis.
    state1, state2 : strings
        The states of the modification site (mod_site) on the substrate before
        (state1) and after (state2) catalysis.
    klist : list of 3 Parameters or list of 3 numbers
        Forward, reverse and catalytic rate constants (in that order). If
        Parameters are passed, they will be used directly in the generated
        Rules. If numbers are passed, Parameters will be created with
        automatically generated names based on the names and states of enzyme,
        substrate and product and these parameters will be included at the end
        of the returned component list.

    Returns
    -------
    components : ComponentSet
        The generated components. Contains two Rules (bidirectional complex
        formation and unidirectional product dissociation), and optionally three
        Parameters if klist was given as plain numbers.

    Notes
    -----
    When passing a MonomerPattern for `enzyme` or `substrate`, do not include
    `e_site` or `s_site` in the respective patterns. In addition, do not
    include the state of the modification site on the substrate. The macro
    will handle this.

    Examples
    --------
    Using a single Monomer for substrate and product with a state change::

        Monomer('Kinase', ['b'])
        Monomer('Substrate', ['b', 'y'], {'y': ('U', 'P')})
        catalyze_state(Kinase, 'b', Substrate, 'b', 'y', 'U', 'P',
                 (1e-4, 1e-1, 1))
    """

    return catalyze(enzyme, e_site, substrate({mod_site: state1}),
                    s_site, substrate({mod_site: state2}), klist)

def catalyze_one_step(enzyme, substrate, product, kf):
    """Generate the one-step catalytic reaction E + S >> E + P.

    Parameters
    ----------
    enzyme, substrate, product : Monomer or MonomerPattern
        E, S and P in the above reaction.
    kf : a Parameter or a number
        Forward rate constant for the reaction. If a
        Parameter is passed, it will be used directly in the generated
        Rules. If a number is passed, a Parameter will be created with an
        automatically generated name based on the names and states of the
        enzyme, substrate and product and this parameter will be included
        at the end of the returned component list.

    Returns
    -------
    components : ComponentSet
        The generated components. Contains the unidirectional reaction Rule
        and optionally the forward rate Parameter if klist was given as a
        number.

    Notes
    -----
    In this macro, there is no direct binding between enzyme and substrate,
    so binding sites do not have to be specified. This represents an
    approximation for the case when the enzyme is operating in its linear
    range. However, if catalysis is nevertheless contingent on the enzyme or
    substrate being unbound on some site, then that information must be encoded
    in the MonomerPattern for the enzyme or substrate. See the examples, below.

    Examples
    --------
        Model()
        Monomer('E', ['b'])
        Monomer('S', ['b'])
        Monomer('P')
        catalyze_one_step(E, S, P, 1e-4)

    If the ability of the enzyme E to catalyze this reaction is dependent
    on the site 'b' of E being unbound, then this macro must be called as

        catalyze_one_step(E(b=None), S, P, 1e-4)

    and similarly if the substrate or product must be unbound.
    """

    return _macro_rule('one_step',
                       enzyme() + substrate() >> enzyme() + product(),
                       [kf], ['kf'])

def catalyze_one_step_reversible(enzyme, substrate, product, klist):
    """Create fwd and reverse rules for catalysis of the form:
       E + S -> E + P
           P -> S 

    Parameters
    ----------
    enzyme, substrate, product : Monomer or MonomerPattern
        E, S and P in the above reactions.
    klist : list of 2 Parameters or list of 2 numbers
        A list containing the rate constant for catalysis and the rate constant
        for the conversion of product back to substrate (in that order). If
        Parameters are passed, they will be used directly in the generated
        Rules. If numbers are passed, Parameters will be created with
        automatically generated names based on the names and states of S1 and
        S2 and these parameters will be included at the end of the returned
        component list.

    Returns
    -------
    components : ComponentSet
        The generated components. Contains two rules (the single-step catalysis
        rule and the product reversion rule) and optionally the two generated
        Parameter objects if klist was given as numbers.

    Notes
    -----
    Calls the macro catalyze_one_step to generate the catalysis rule.

    Examples
    --------
        Model()
        Monomer('E', ['b'])
        Monomer('S', ['b'])
        Monomer('P')
        catalyze_one_step_reversible(E, S, P, [1e-1, 1e-4])
    """

    components = catalyze_one_step(enzyme, substrate, product, klist[0])

    components |= _macro_rule('reverse', product() >> substrate(),
                              [klist[1]], ['kr'])
    return components

# Synthesis and degradation
# =========================

def synthesize(species, ksynth):
    """Generate a reaction which synthesizes a species.

    Note that `species` must be "concrete", i.e. the state of all
    sites in all of its monomers must be specified. No site may be
    left unmentioned.

    Parameters
    ----------
    species : Monomer, MonomerPattern or ComplexPattern
        The species to synthesize. If a Monomer, sites are considered
        as unbound and in their default state. If a pattern, must be
        concrete.
    ksynth : Parameters or number
        Synthesis rate. If a Parameter is passed, it will be used directly in
        the generated Rule. If a number is passed, a Parameter will be created
        with an automatically generated name based on the names and site states
        of the components of `species` and this parameter will be included at
        the end of the returned component list.

    Returns
    -------
    components : ComponentSet
        The generated components. Contains the unidirectional synthesis Rule and
        optionally a Parameter if ksynth was given as a number.

    Examples
    --------
        Model()
        Monomer('A', ['x', 'y'], {'y': ['e', 'f']})
        synthesize(A(x=None, y='e'), 1e-4)

    """

    def synthesize_name_func(rule_expression):
        cps = rule_expression.product_pattern.complex_patterns
        return '_'.join(_complex_pattern_label(cp) for cp in cps)

    if isinstance(species, Monomer):
        species = species()
    species = as_complex_pattern(species)
    if not species.is_concrete():
        raise ValueError("species must be concrete")

    return _macro_rule('synthesize', None >> species, [ksynth], ['k'],
                       name_func=synthesize_name_func)

def degrade(species, kdeg):
    """Generate a reaction which degrades a species.

    Note that `species` is not required to be "concrete".

    Parameters
    ----------
    species : Monomer, MonomerPattern or ComplexPattern
        The species to synthesize. If a Monomer, sites are considered
        as unbound and in their default state. If a pattern, must be
        concrete.
    kdeg : Parameters or number
        Degradation rate. If a Parameter is passed, it will be used directly in
        the generated Rule. If a number is passed, a Parameter will be created
        with an automatically generated name based on the names and site states
        of the components of `species` and this parameter will be included at
        the end of the returned component list.

    Returns
    -------
    components : ComponentSet
        The generated components. Contains the unidirectional degradation Rule
        and optionally a Parameter if ksynth was given as a number.

    Examples
    --------
        Model()
        Monomer('B', ['x'])
        degrade(B(), 1e-6)  # degrade all B, even bound species

    """

    def degrade_name_func(rule_expression):
        cps = rule_expression.reactant_pattern.complex_patterns
        return '_'.join(_complex_pattern_label(cp) for cp in cps)

    if isinstance(species, Monomer):
        species = species()
    species = as_complex_pattern(species)

    return _macro_rule('degrade', species >> None, [kdeg], ['k'],
                       name_func=degrade_name_func)

def synthesize_degrade_table(table):
    """Generate a table of synthesis and degradation reactions.

    Given a list of species, calls the `synthesize` and `degrade` macros on each
    one. The species and the parameter values are passed as a list of lists
    (i.e. a table) with each inner list consisting of the species, forward and
    reverse rates (in that order).

    Each species' associated pair of rates may be either Parameters or
    numbers. If Parameters are passed, they will be used directly in the
    generated Rules. If numbers are passed, Parameters will be created with
    automatically generated names based on the names and states of the relevant
    species and these parameters will be included in the returned component
    list. To omit any individual reaction, pass None in place of the
    corresponding parameter.

    Note that any `species` with a non-None synthesis rate must be "concrete".

    Parameters
    ----------
    table : list of lists
        Table of species and rates, as described above.

    Returns
    -------
    components : ComponentSet
        The generated components. Contains the unidirectional synthesis and
        degradation Rules and optionally the Parameters for any rates given as
        numbers.

    Examples
    --------
        Model()
        Monomer('A', ['x', 'y'], {'y': ['e', 'f']})
        Monomer('B', ['x'])
        synthesize_degrade_table([[A(x=None, y='e'), 1e-4, 1e-6],
                                  [B(),              None, 1e-7]])

    """

    # loop over interactions
    components = ComponentSet()
    for row in table:
        species, ksynth, kdeg = row
        if ksynth is not None:
            components |= synthesize(species, ksynth)
        if kdeg is not None:
            components |= degrade(species, kdeg)

    return components

# Pore assembly
# =============

def pore_species(subunit, site1, site2, size):
    """Return a MonomerPattern representing a circular homomeric pore.

    Parameters
    ----------
    subunit : Monomer or MonomerPattern
        The subunit of which the pore is composed.
    site1, site2 : string
        The names of the sites where one copy of `subunit` binds to the next.
    size : integer
        The number of subunits in the pore.

    Returns
    -------
    A MonomerPattern corresponding to the pore.

    Notes
    -----
    At sizes 1 and 2 the ring is not closed, i.e. there is one site1 and one
    site2 which remain unbound. At size 3 and up the ring is closed and all
    site1 sites are bound to a site2.

    Examples
    --------
        Model()
        Monomer('Unit', ['p1', 'p2'])
        pore_tetramer = pore_species(Unit, 'p1', 'p2', 4)

    """
    _verify_sites(subunit, site1, site2)
    if size <= 0:
        raise ValueError("size must be an integer greater than 0")
    if size == 1:
        pore = subunit({site1: None, site2: None})
    elif size == 2:
        pore = subunit({site1: 1, site2: None}) % \
               subunit({site1: None, site2: 1})
    else:
        # build up a ComplexPattern, starting with a single subunit
        pore = subunit({site1: 1, site2: 2})
        for i in range(2, size + 1):
            pore %= subunit({site1: i, site2: i % size + 1})
        pore.match_once = True
    return pore

def assemble_pore_sequential(subunit, site1, site2, max_size, ktable):
    """Generate rules to assemble a circular homomeric pore sequentially.

    The pore species are created by sequential addition of `subunit` monomers,
    i.e. larger oligomeric species never fuse together. The pore structure is
    defined by the `pore_species` macro.

    Parameters
    ----------
    subunit : Monomer or MonomerPattern
        The subunit of which the pore is composed.
    site1, site2 : string
        The names of the sites where one copy of `subunit` binds to the next.
    max_size : integer
        The maximum number of subunits in the pore.
    ktable : list of lists of Parameters or numbers
        Table of forward and reverse rate constants for the assembly steps. The
        outer list must be of length `max_size` - 1, and the inner lists must
        all be of length 2. In the outer list, the first element corresponds to
        the first assembly step in which two monomeric subunits bind to form a
        2-subunit complex, and the last element corresponds to the final step in
        which the `max_size`th subunit is added. Each inner list contains the
        forward and reverse rate constants (in that order) for the corresponding
        assembly reaction, and each of these pairs must comprise solely
        Parameter objects or solely numbers (never one of each). If Parameters
        are passed, they will be used directly in the generated Rules. If
        numbers are passed, Parameters will be created with automatically
        generated names based on `subunit`, `site1`, `site2` and the pore sizes
        and these parameters will be included at the end of the returned
        component list.

    """
    if len(ktable) != max_size - 1:
        raise ValueError("len(ktable) must be equal to max_size - 1")

    def pore_rule_name(rule_expression, size):
        react_p = rule_expression.reactant_pattern
        monomer = react_p.complex_patterns[0].monomer_patterns[0].monomer
        return '%s_%d' % (monomer.name, size)

    components = ComponentSet()
    s = pore_species(subunit, site1, site2, 1)
    for size, klist in zip(range(2, max_size + 1), ktable):
        pore_prev = pore_species(subunit, site1, site2, size - 1)
        pore_next = pore_species(subunit, site1, site2, size)
        name_func = functools.partial(pore_rule_name, size=size)
        components |= _macro_rule('assemble_pore_sequential',
                                  s + pore_prev <> pore_next,
                                  klist, ['kf', 'kr'],
                                  name_func=name_func)

    return components

def pore_transport(subunit, sp_site1, sp_site2, sc_site, min_size, max_size,
                   csource, c_site, cdest, ktable):
    """Generate rules to transport cargo through a circular homomeric pore.

    The pore structure is defined by the `pore_species` macro -- `subunit`
    monomers bind to each other from `sp_site1` to `sp_site2` to form a closed
    ring. The transport reaction is modeled as a catalytic process of the form
    pore + csource <> pore:csource >> pore + cdest

    Parameters
    ----------
    subunit : Monomer or MonomerPattern
        Subunit of which the pore is composed.
    sp_site1, sp_site2 : string
        Names of the sites where one copy of `subunit` binds to the next.
    sc_site : string
        Name of the site on `subunit` where it binds to the cargo `csource`.
    min_size, max_size : integer
        Minimum and maximum number of subunits in the pore at which transport
        will occur.
    csource : Monomer or MonomerPattern
        Cargo "source", i.e. the entity to be transported.
    c_site : string
        Name of the site on `csource` where it binds to `subunit`.
    cdest : Monomer or MonomerPattern
        Cargo "destination", i.e. the resulting state after the transport event.
    ktable : list of lists of Parameters or numbers
        Table of forward, reverse and catalytic rate constants for the transport
        reactions. The outer list must be of length `max_size` - `min_size` + 1,
        and the inner lists must all be of length 3. In the outer list, the
        first element corresponds to the transport through the pore of size
        `min_size` and the last element to that of size `max_size`. Each inner
        list contains the forward, reverse and catalytic rate constants (in that
        order) for the corresponding transport reaction, and each of these pairs
        must comprise solely Parameter objects or solely numbers (never some of
        each). If Parameters are passed, they will be used directly in the
        generated Rules. If numbers are passed, Parameters will be created with
        automatically generated names based on the subunit, the pore size and
        the cargo, and these parameters will be included at the end of the
        returned component list.
    """

    _verify_sites(subunit, sc_site)
    _verify_sites(csource, c_site)

    if len(ktable) != max_size - min_size + 1:
        raise ValueError("len(ktable) must be equal to max_size - min_size + 1")

    def pore_transport_rule_name(rule_expression, size):
        # Get ReactionPatterns
        react_p = rule_expression.reactant_pattern
        prod_p = rule_expression.product_pattern
        # Build the label components
        # Pore is always first complex of LHS due to how we build the rules
        subunit = react_p.complex_patterns[0].monomer_patterns[0]
        if len(react_p.complex_patterns) == 2:
            # This is the complexation reaction
            cargo = react_p.complex_patterns[1].monomer_patterns[0]
        else:
            # This is the dissociation reaction
            cargo = prod_p.complex_patterns[1].monomer_patterns[0]
        return '%s_%d_%s' % (_monomer_pattern_label(subunit), size,
                             _monomer_pattern_label(cargo))

    components = ComponentSet()
    # Set up some aliases that are invariant with pore size
    subunit_free = subunit({sc_site: None})
    csource_free = csource({c_site: None})
    # If cdest is actually a variant of csource, we need to explicitly say that
    # it is no longer bound to the pore
    if cdest().monomer is csource().monomer:
        cdest = cdest({c_site: None})

    for size, klist in zip(range(min_size, max_size + 1), ktable):
        # More aliases which do depend on pore size
        pore_free = pore_species(subunit_free, sp_site1, sp_site2, size)

        # This one is a bit tricky. The pore:csource complex must only introduce
        # one additional bond even though there are multiple subunits in the
        # pore. We create partial patterns for bound pore and csource, using a
        # bond number that is high enough not to conflict with the bonds within
        # the pore ring itself.
        # Start by copying pore_free, which has all cargo binding sites empty
        pore_bound = pore_free.copy()
        # Get the next bond number not yet used in the pore structure itself
        cargo_bond_num = size + 1
        # Assign that bond to the first subunit in the pore
        pore_bound.monomer_patterns[0].site_conditions[sc_site] = cargo_bond_num
        # Create a cargo source pattern with that same bond
        csource_bound = csource({c_site: cargo_bond_num})
        # Finally we can define the complex trivially; the bond numbers are
        # already present in the patterns
        pc_complex = pore_bound % csource_bound

        # Create the rules (just like catalyze)
        name_func = functools.partial(pore_transport_rule_name, size=size)
        components |= _macro_rule('pore_transport_complex',
                                  pore_free + csource_free <> pc_complex,
                                  klist[0:2], ['kf', 'kr'],
                                  name_func=name_func)
        components |= _macro_rule('pore_transport_dissociate',
                                  pc_complex >> pore_free + cdest,
                                  [klist[2]], ['kc'],
                                  name_func=name_func)

    return components

def pore_bind(subunit, sp_site1, sp_site2, sc_site, size, cargo, c_site,
              klist):
    """Generate rules to bind a monomer to a circular homomeric pore.

    The pore structure is defined by the `pore_species` macro -- `subunit`
    monomers bind to each other from `sp_site1` to `sp_site2` to form a closed
    ring. The binding reaction takes the form pore + cargo <> pore:cargo.

    Parameters
    ----------
    subunit : Monomer or MonomerPattern
        Subunit of which the pore is composed.
    sp_site1, sp_site2 : string
        Names of the sites where one copy of `subunit` binds to the next.
    sc_site : string
        Name of the site on `subunit` where it binds to the cargo `cargo`.
    size : integer
        Number of subunits in the pore at which binding will occur.
    cargo : Monomer or MonomerPattern
        Cargo that binds to the pore complex.
    c_site : string
        Name of the site on `cargo` where it binds to `subunit`.
    klist : list of Parameters or numbers
        List containing forward and reverse rate constants for the binding
        reaction (in that order). Rate constants should either be both Parameter
        objects or both numbers. If Parameters are passed, they will be used
        directly in the generated Rules. If numbers are passed, Parameters
        will be created with automatically generated names based on the
        subunit, the pore size and the cargo, and these parameters will be
        included at the end of the returned component list.
    """

    _verify_sites(subunit, sc_site)
    _verify_sites(cargo, c_site)

    def pore_bind_rule_name(rule_expression, size):
        # Get ReactionPatterns
        react_p = rule_expression.reactant_pattern
        prod_p = rule_expression.product_pattern
        # Build the label components
        # Pore is always first complex of LHS due to how we build the rules
        subunit = react_p.complex_patterns[0].monomer_patterns[0].monomer
        if len(react_p.complex_patterns) == 2:
            # This is the complexation reaction
            cargo = react_p.complex_patterns[1].monomer_patterns[0]
        else:
            # This is the dissociation reaction
            cargo = prod_p.complex_patterns[1].monomer_patterns[0]
        return '%s_%d_%s' % (subunit.name, size,
                             _monomer_pattern_label(cargo))

    components = ComponentSet()
    # Set up some aliases that are invariant with pore size
    subunit_free = subunit({sc_site: None})
    cargo_free = cargo({c_site: None})

    #for size, klist in zip(range(min_size, max_size + 1), ktable):

    # More aliases which do depend on pore size
    pore_free = pore_species(subunit_free, sp_site1, sp_site2, size)

    # This one is a bit tricky. The pore:cargo complex must only introduce
    # one additional bond even though there are multiple subunits in the
    # pore. We create partial patterns for bound pore and cargo, using a
    # bond number that is high enough not to conflict with the bonds within
    # the pore ring itself.
    # Start by copying pore_free, which has all cargo binding sites empty
    pore_bound = pore_free.copy()
    # Get the next bond number not yet used in the pore structure itself
    cargo_bond_num = size + 1
    # Assign that bond to the first subunit in the pore
    pore_bound.monomer_patterns[0].site_conditions[sc_site] = cargo_bond_num
    # Create a cargo source pattern with that same bond
    cargo_bound = cargo({c_site: cargo_bond_num})
    # Finally we can define the complex trivially; the bond numbers are
    # already present in the patterns
    pc_complex = pore_bound % cargo_bound

    # Create the rules
    name_func = functools.partial(pore_bind_rule_name, size=size)
    components |= _macro_rule('pore_bind',
                              pore_free + cargo_free <> pc_complex,
                              klist[0:2], ['kf', 'kr'],
                              name_func=name_func)

    return components

