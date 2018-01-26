"""
A collection of generally useful modeling macros.

These macros are written to be as generic and reusable as possible, serving as a
collection of best practices and implementation ideas. They conform to the
following general guidelines:

* All components created by the macro are implicitly added to the current model
  and explicitly returned in a ComponentSet.

* Parameters may be passed as Parameter or Expression objects, or as plain
  numbers for which Parameter objects will be automatically created using an
  appropriate naming convention.

* Arguments which accept a MonomerPattern should also accept Monomers, which are
  to be interpreted as MonomerPatterns on that Monomer with an empty condition
  list. This is typically implemented by having the macro apply the "call"
  (parentheses) operator to the argument with an empty argument list and using
  the resulting value instead of the original argument when creating Rules, e.g.
  ``arg = arg()``. Calling a Monomer will return a MonomerPattern, and calling a
  MonomerPattern will return a copy of itself, so calling either is guaranteed
  to return a MonomerPattern.

The _macro_rule helper function contains much of the logic needed to follow
these guidelines. Every macro in this module either uses _macro_rule directly or
calls another macro which does.

Another useful function is _verify_sites which will raise an exception if a
Monomer or MonomerPattern does not possess every one of a given list of sites.
This can be used to trigger such errors up front rather than letting an
exception occur at the point where the macro tries to use the invalid site in a
pattern, which can be harder for the caller to debug.
"""


import inspect
from pysb import *
import pysb.core
from pysb.core import ComponentSet, as_reaction_pattern, as_complex_pattern, MonomerPattern, ComplexPattern
import numbers
import functools
import itertools

__all__ = ['equilibrate',
           'bind', 'bind_table',
           'catalyze', 'catalyze_state', 'catalyze_complex',
           'catalyze_one_step', 'catalyze_one_step_reversible',
           'synthesize', 'degrade', 'synthesize_degrade_table',
           'assemble_pore_sequential', 'pore_transport', 'pore_bind', 'assemble_chain_sequential_base',
           'bind_complex', 'bind_table_complex']

# Suppress ModelExistsWarnings in our doctests.
_pysb_doctest_suppress_modelexistswarning = True


# Internal helper functions
# =========================

def _complex_pattern_label(cp):
    """Return a string label for a ComplexPattern."""
    if cp is None:
        return ''
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
    """
    A helper function for writing macros that generates a single rule.

    Parameters
    ----------
    rule_prefix : string
        The prefix that is prepended to the (automatically generated) name for
        the rule.
    rule_expression : RuleExpression
        An expression specifying the form of the rule; gets passed directly
        to the Rule constructor.
    klist : list of Parameters or Expressions, or list of numbers
        If the rule is unidirectional, the list must contain one element
        (either a Parameter/Expression or number); if the rule is reversible,
        it must contain two elements. If the rule is reversible, the first
        element in the list is taken to be the forward rate, and the second
        element is taken as the reverse rate. 
    ksuffixes : list of strings
        If klist contains numbers rather than Parameters or Expressions, the
        strings in ksuffixes are used to automatically generate the necessary
        Parameter objects. The suffixes are appended to the rule name to
        generate the associated parameter name. ksuffixes must contain one
        element if the rule is unidirectional, two if it is reversible.
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
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('A', ['s'])
        Monomer('A', ['s'])
        >>> Monomer('B', ['s'])
        Monomer('B', ['s'])
        >>> 
        >>> _macro_rule('bind', A(s=None) + B(s=None) | A(s=1) % B(s=1),
        ... [1e6, 1e-1], ['kf', 'kr']) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('bind_A_B_to_AB', A(s=None) + B(s=None) | A(s=1) % B(s=1),
             bind_A_B_to_AB_kf, bind_A_B_to_AB_kr),
         Parameter('bind_A_B_to_AB_kf', 1000000.0),
         Parameter('bind_A_B_to_AB_kr', 0.1),
         ])

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

    if all(isinstance(x, (Parameter, Expression)) for x in klist):
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
        raise ValueError("klist must contain Parameters, Expressions, or numbers.")

    if rule_expression.is_reversible:
        r = Rule(r_name, rule_expression, k1, k2)
    else:
        r = Rule(r_name, rule_expression, k1)

    # Build a set of components that were created
    return ComponentSet([r]) | params_created

def _verify_sites(m, *site_list):
    """
    Checks that the monomer m contains all of the sites in site_list.

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

    if isinstance(m, ComplexPattern):
        return _verify_sites_complex(m, *site_list)
    else:
        for site in site_list:
            if site not in m().monomer.sites:
                raise ValueError("Monomer '%s' must contain the site '%s'" %
                                (m().monomer.name, site))
        return True

def _verify_sites_complex(c, *site_list):
  
    """
    Checks that the complex c contains all of the sites in site_list.

    Parameters
    ----------
    c : ComplexPattern
        The complex to check.
    site1, site2, ... : string
        One or more site names to check on c

    Returns
    -------
    If all sites are found within the complex, a dictionary of monomers and the sites within site_list they contain.  Raises a ValueError if one or more sites not in the complex.

    Raises
    ------
    ValueError
         If any of the sites are not found within the complex.

    """

    allsitesdict = {}
    for mon in c.monomer_patterns:
        allsitesdict[mon] = mon.monomer.sites
    for site in site_list:
        specsitesdict = {}
        for monomer, li in allsitesdict.items():
            for s in li:
                if site in li:
                    specsitesdict[monomer] = site
        if len(specsitesdict) == 0:
            raise ValueError("Site '%s' not found in complex '%s'" % (site, c))
    return specsitesdict

# Unimolecular patterns
# =====================

def equilibrate(s1, s2, klist):
    """
    Generate the unimolecular reversible equilibrium reaction S1 <-> S2.

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

    Examples
    --------
    Simple two-state equilibrium between A and B::

        Model()
        Monomer('A')
        Monomer('B')
        equilibrate(A(), B(), [1, 1])
    
    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('A')
        Monomer('A')
        >>> Monomer('B')
        Monomer('B')
        >>> equilibrate(A(), B(), [1, 1]) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('equilibrate_A_to_B', A() | B(), equilibrate_A_to_B_kf, equilibrate_A_to_B_kr),
         Parameter('equilibrate_A_to_B_kf', 1.0),
         Parameter('equilibrate_A_to_B_kr', 1.0),
         ])

    """
    
    # turn any Monomers into MonomerPatterns
    return _macro_rule('equilibrate', s1 | s2, klist, ['kf', 'kr'])

# Binding
# =======

def bind(s1, site1, s2, site2, klist):
    """
    Generate the reversible binding reaction S1 + S2 | S1:S2.

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
    Binding between A and B::

        Model()
        Monomer('A', ['x'])
        Monomer('B', ['y'])
        bind(A, 'x', B, 'y', [1e-4, 1e-1])

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('A', ['x'])
        Monomer('A', ['x'])
        >>> Monomer('B', ['y'])
        Monomer('B', ['y'])
        >>> bind(A, 'x', B, 'y', [1e-4, 1e-1]) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('bind_A_B', A(x=None) + B(y=None) | A(x=1) % B(y=1), bind_A_B_kf, bind_A_B_kr),
         Parameter('bind_A_B_kf', 0.0001),
         Parameter('bind_A_B_kr', 0.1),
         ])

    """

    _verify_sites(s1, site1)
    _verify_sites(s2, site2)
    return _macro_rule('bind',
                       s1(**{site1: None}) + s2(**{site2: None}) |
                       s1(**{site1: 1}) % s2(**{site2: 1}),
                       klist, ['kf', 'kr'], name_func=bind_name_func)

def bind_name_func(rule_expression):
    # Get ComplexPatterns
    react_cps = rule_expression.reactant_pattern.complex_patterns
    # Build the label components
    return '_'.join(_complex_pattern_label(cp) for cp in react_cps)


def bind_complex(s1, site1, s2, site2, klist, m1=None, m2=None):
    """
    Generate the reversible binding reaction ``S1 + S2 | S1:S2``,
    with optional complexes attached to either
    ``S1`` (``C1:S1 + S2 | C1:S1:S2``), ``S2`` (``S1 + C2:S2 | C2:S2:S1``),
    or both (``C1:S1 + C2:S2 | C1:S1:S2:C2``).

    Parameters
    ----------
    s1, s2 : Monomer, MonomerPattern, or ComplexPattern
        Monomers or complexes participating in the binding reaction.
    site1, site2 : string
        The names of the sites on s1 and s2 used for binding.
    klist : list of 2 Parameters or list of 2 numbers
        Forward and reverse rate constants (in that order). If Parameters are
        passed, they will be used directly in the generated Rules. If numbers
        are passed, Parameters will be created with automatically generated
        names based on the names and states of S1 and S2 and these parameters
        will be included at the end of the returned component list.
    m1, m2 : Monomer or MonomerPattern
        If s1 or s2 binding site is present in multiple monomers
        within a complex, the specific monomer desired for binding must be specified.

    Returns
    -------
    components : ComponentSet
        The generated components. Contains the bidirectional binding Rule
        and optionally two Parameters if klist was given as numbers.

    Examples
    --------

    Binding between ``A:B`` and ``C:D``:

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' ...>
        >>> Monomer('A', ['a', 'b'])
        Monomer('A', ['a', 'b'])
        >>> Monomer('B', ['c', 'd'])
        Monomer('B', ['c', 'd'])
        >>> Monomer('C', ['e', 'f'])
        Monomer('C', ['e', 'f'])
        >>> Monomer('D', ['g', 'h'])
        Monomer('D', ['g', 'h'])
        >>> bind_complex(A(a=1) % B(c=1), 'b', C(e=2) % D(g=2), 'h', [1e-4, \
            1e-1]) #doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
        Rule('bind_AB_DC', A(a=1, b=None) % B(c=1) + D(g=3, h=None) % C(e=3)
          | A(a=1, b=50) % B(c=1) % D(g=3, h=50) % C(e=3), bind_AB_DC_kf,
          bind_AB_DC_kr),
        Parameter('bind_AB_DC_kf', 0.0001),
        Parameter('bind_AB_DC_kr', 0.1),
        ])

    Execution:

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' ...>
        >>> Monomer('A', ['a', 'b'])
        Monomer('A', ['a', 'b'])
        >>> Monomer('B', ['c', 'd'])
        Monomer('B', ['c', 'd'])
        >>> Monomer('C', ['e', 'f'])
        Monomer('C', ['e', 'f'])
        >>> Monomer('D', ['g', 'h'])
        Monomer('D', ['g', 'h'])
        >>> bind(A, 'a', B, 'c', [1e4, 1e-1]) #doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
        Rule('bind_A_B',
          A(a=None) + B(c=None) | A(a=1) % B(c=1),
          bind_A_B_kf, bind_A_B_kr),
        Parameter('bind_A_B_kf', 10000.0),
        Parameter('bind_A_B_kr', 0.1),
        ])
        >>> bind(C, 'e', D, 'g', [1e4, 1e-1]) #doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
        Rule('bind_C_D',
          C(e=None) + D(g=None) | C(e=1) % D(g=1),
          bind_C_D_kf, bind_C_D_kr),
        Parameter('bind_C_D_kf', 10000.0),
        Parameter('bind_C_D_kr', 0.1),
        ])
        >>> bind_complex(A(a=1) % B(c=1), 'b', C(e=2) % D(g=2), 'h', [1e-4, \
            1e-1]) #doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
        Rule('bind_AB_DC',
          A(a=1, b=None) % B(c=1) + D(g=3, h=None) % C(e=3) | A(a=1,
          b=50) % B(c=1) % D(g=3, h=50) % C(e=3),
          bind_AB_DC_kf, bind_AB_DC_kr),
        Parameter('bind_AB_DC_kf', 0.0001),
        Parameter('bind_AB_DC_kr', 0.1),
        ])
    """
    if isinstance(m1, Monomer):
        m1 = m1()
    if isinstance(m2, Monomer):
        m2 = m2()
    #Define some functions for checking complex sites, building complexes up from monomers, and creating rules.
    def comp_mono_func(s1, site1, s2, site2, m1):
        _verify_sites(s2, site2)
        #Retrieve a dictionary specifying the MonomerPattern within the complex that contains the given binding site.
        specsites = list(_verify_sites_complex(s1, site1))
        s1complexpatub, s1complexpatb = check_sites_comp_build(s1, site1, m1, specsites)
        return create_rule(s1complexpatub, s1complexpatb, s2({site2:None}), s2({site2: 50}))

    def check_sites_comp_build(s1, site1, m1, specsites):
        #Return error if binding site exists on multiple monomers and a monomer for binding (m1) hasn't been specified.
        if len(specsites) > 1 and m1==None:
            raise ValueError("Binding site '%s' present in more than one monomer in complex '%s'.  Specify variable m1, the monomer used for binding within the complex." % (site1, s1))
        if not s1.is_concrete:
            raise ValueError("Complex '%s' must be concrete." % (s1))
        #If the given binding site is only present in one monomer in the complex:
        if m1==None:
            #Build up ComplexPattern for use in rule (with state of given binding site specified).
            s1complexpatub = specsites[0]({site1:None})
            s1complexpatb = specsites[0]({site1:50})
            for monomer in s1.monomer_patterns:
                if monomer not in specsites:
                    s1complexpatub %= monomer
                    s1complexpatb %= monomer

        #If the binding site is present on more than one monomer in the complex, the monomer must be specified by the user.  Use specified m1 to build ComplexPattern.
        else:
            #Make sure binding states of MonomerPattern m1 match those of the monomer within the ComplexPattern s1 (ComplexPattern monomer takes precedence if not).
            i = 0
            identical_monomers = []
            other_monomers = []
            for mon in s1.monomer_patterns:
                #Only change the binding site for the first monomer that matches.  Keep any others unchanged to add to final complex that is returned.
                if mon.monomer.name == m1.monomer.name and mon.site_conditions==m1.site_conditions:
                    i += 1
                    if i == 1:
                        s1complexpatub = mon({site1:None})
                        s1complexpatb = mon({site1:50})
                    else:
                        identical_monomers.append(mon)
                else:
                    other_monomers.append(mon)

            #Throw an error if no monomer pattern in the complex matched the pattern given for m1
            if i == 0:
                raise ValueError("No monomer pattern in complex '%s' matches the pattern given for m1, '%s'." % (s1, m1))

            #Build up ComplexPattern for use in rule (with state of given binding site on m1 specified).
            for mon in other_monomers:
                    s1complexpatub %= mon
                    s1complexpatb %= mon
            if identical_monomers:
                for i in range(len(identical_monomers)):
                    s1complexpatub %= identical_monomers[i]
                    s1complexpatb %= identical_monomers[i]

        return s1complexpatub, s1complexpatb
    #Create rules.
    def create_rule(s1ub, s1b, s2ub, s2b):
        return _macro_rule('bind',
                            s1ub + s2ub |
                            s1b % s2b,
                            klist, ['kf', 'kr'], name_func=bind_name_func)

    #If no complexes given, revert to normal bind macro.
    if (isinstance(s1, MonomerPattern) or isinstance(s1, Monomer)) and (isinstance(s2, MonomerPattern) or isinstance(s2, Monomer)):
        _verify_sites(s1, site1)
        _verify_sites(s2, site2)
        return bind(s1, site1, s2, site2, klist)

    #Create rules if only one complex or the other is present.
    elif isinstance(s1, ComplexPattern) and (isinstance(s2, MonomerPattern) or isinstance(s2, Monomer)):
        return comp_mono_func(s1, site1, s2, site2, m1)
    elif (isinstance(s1, MonomerPattern) or isinstance(s1, Monomer)) and isinstance(s2, ComplexPattern):
        return comp_mono_func(s2, site2, s1, site1, m2)

    #Create rule when both s1 and s2 are complexes.
    else:
        #Retrieve a dictionary specifiying the MonomerPattern within
        #the complex that contains the given binding site. Convert to list.
        specsites1 = list(_verify_sites_complex(s1, site1))
        specsites2 = list(_verify_sites_complex(s2, site2))
        #Return error if binding site exists on multiple monomers and a monomer for binding (m1/m2) hasn't been specified.
        if len(specsites1) > 1 and m1==None:
            raise ValueError("Binding site '%s' present in more than one monomer in complex '%s'.  Specify variable m1, the monomer used for binding within the complex." % (site1, s1))
        if len(specsites2) > 1 and m2==None:
            raise ValueError("Binding site '%s' present in more than one monomer in complex '%s'.  Specify variable m2, the monomer used for binding within the complex." % (site2, s2))
        if not s1.is_concrete:
            raise ValueError("Complex '%s' must be concrete." % (s1))
        if not s2.is_concrete:
            raise ValueError("Complex '%s' must be concrete." % (s2))
        #To avoid creating rules with multiple bonds to the same site when combining the two complexes, check for the maximum bond integer in s1 and add to all s2 bond integers.
        maxint = 0
        for monomer in s1.monomer_patterns:
            for stateint in monomer.site_conditions.values():
                if isinstance(stateint, int):
                    if stateint > maxint:
                        maxint = stateint
        match = 'N'
        for monomer in s2.monomer_patterns:
            if m2 is not None:
                if m2.site_conditions == monomer.site_conditions and m2.monomer.name == monomer.monomer.name:
                    match = 'Y'
            for site, stateint in monomer.site_conditions.items():
                if isinstance(stateint, int):
                    monomer.site_conditions[site] += maxint
            if match == 'Y':
                m2.site_conditions = monomer.site_conditions
            match = 'N'

        #Actually create rules
        s1complexpatub, s1complexpatb = check_sites_comp_build(s1, site1, m1, specsites1)
        s2complexpatub, s2complexpatb = check_sites_comp_build(s2, site2, m2, specsites2)
        return create_rule(s1complexpatub, s1complexpatb, s2complexpatub, s2complexpatb)


def bind_table(bindtable, row_site, col_site, kf=None):
    """
    Generate a table of reversible binding reactions.

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
    Binding table for two species types (R and C), each with two members::

        Model()
        Monomer('R1', ['x'])
        Monomer('R2', ['x'])
        Monomer('C1', ['y'])
        Monomer('C2', ['y'])
        bind_table([[               C1,           C2],
                    [R1,  (1e-4, 1e-1),  (2e-4, 2e-1)],
                    [R2,  (3e-4, 3e-1),         None]],
                   'x', 'y')

    Execution:: 

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('R1', ['x'])
        Monomer('R1', ['x'])
        >>> Monomer('R2', ['x'])
        Monomer('R2', ['x'])
        >>> Monomer('C1', ['y'])
        Monomer('C1', ['y'])
        >>> Monomer('C2', ['y'])
        Monomer('C2', ['y'])
        >>> bind_table([[               C1,           C2],
        ...             [R1,  (1e-4, 1e-1),  (2e-4, 2e-1)],
        ...             [R2,  (3e-4, 3e-1),         None]],
        ...            'x', 'y') # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('bind_R1_C1', R1(x=None) + C1(y=None) | R1(x=1) % C1(y=1),
             bind_R1_C1_kf, bind_R1_C1_kr),
         Parameter('bind_R1_C1_kf', 0.0001),
         Parameter('bind_R1_C1_kr', 0.1),
         Rule('bind_R1_C2', R1(x=None) + C2(y=None) | R1(x=1) % C2(y=1),
             bind_R1_C2_kf, bind_R1_C2_kr),
         Parameter('bind_R1_C2_kf', 0.0002),
         Parameter('bind_R1_C2_kr', 0.2),
         Rule('bind_R2_C1', R2(x=None) + C1(y=None) | R2(x=1) % C1(y=1),
             bind_R2_C1_kf, bind_R2_C1_kr),
         Parameter('bind_R2_C1_kf', 0.0003),
         Parameter('bind_R2_C1_kr', 0.3),
         ])

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


def bind_table_complex(bindtable, row_site, col_site, m1=None, m2=None, kf=None):
    """
    Generate a table of reversible binding reactions when either the row or column species (or both) have a complex bound to them.

    Given two lists of species R and C (which can be complexes or monomers),
    calls the `bind_complex` macro on each pairwise
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
    m1 : Monomer or MonomerPattern, optional
        Monomer in row complex for binding.  Must be specified if there are multiple monomers 
        that have the row_site within a complex.
    m2 : Monomer or MonomerPattern, optional
        Monomer in column complex for binding.  Must be specified if there are multiple monomers 
        that have the col_site within a complex.
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
    Binding table for two species types (R and C, which can be complexes or monomers)::

        Model()
        Monomer('R1', ['x', 'c1'])
        Monomer('R2', ['x', 'c1'])
        Monomer('C1', ['y', 'c2'])
        Monomer('C2', ['y', 'c2'])
        bind(C1(y=None), 'c2', C1(y=None), 'c2', (1e-3, 1e-2))
        bind(R1(x=None), 'c1', R2(x=None), 'c1', (1e-3, 1e-2))
        bind_table_complex([[              C1(c2=1, y=None)%C1(c2=1),            C2],
                           [R1()%R2(),  (1e-4, 1e-1),  (2e-4, 2e-1)],
                           [R2,     (3e-4, 3e-1),         None]],
                           'x', 'y', m1=R1(), m2=C1(y=None, c2=1))

    Execution:: 

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('R1', ['x', 'c1'])
        Monomer('R1', ['x', 'c1'])
        >>> Monomer('R2', ['x', 'c1'])
        Monomer('R2', ['x', 'c1'])
        >>> Monomer('C1', ['y', 'c2'])
        Monomer('C1', ['y', 'c2'])
        >>> Monomer('C2', ['y', 'c2'])
        Monomer('C2', ['y', 'c2'])
        >>> bind(C1(y=None), 'c2', C1(y=None), 'c2', (1e-3, 1e-2)) #doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('bind_C1_C1', C1(y=None, c2=None) + C1(y=None, c2=None) | C1(y=None, c2=1) % C1(y=None, c2=1), bind_C1_C1_kf, bind_C1_C1_kr),
         Parameter('bind_C1_C1_kf', 0.001),
         Parameter('bind_C1_C1_kr', 0.01),
         ])
        >>> bind(R1(x=None), 'c1', R2(x=None), 'c1', (1e-3, 1e-2)) #doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('bind_R1_R2', R1(x=None, c1=None) + R2(x=None, c1=None) | R1(x=None, c1=1) % R2(x=None, c1=1), bind_R1_R2_kf, bind_R1_R2_kr),
         Parameter('bind_R1_R2_kf', 0.001),
         Parameter('bind_R1_R2_kr', 0.01),
         ])
        >>> bind_table_complex([[               C1(c2=1, y=None)%C1(c2=1),           C2],
        ...                      [R1()%R2(),      (1e-4, 1e-1),                        (2e-4, 2e-1)],
        ...                       [R2,             (3e-4, 3e-1),                        None]],
        ...                       'x', 'y', m1=R1(), m2=C1(y=None, c2=1)) #doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
        Rule('bind_R1R2_C1C1', R1(x=None) % R2() + C1(y=None, c2=1) % C1(c2=1) | R1(x=50) % R2() % C1(y=50, c2=1) % C1(c2=1), bind_R1R2_C1C1_kf, bind_R1R2_C1C1_kr),
        Parameter('bind_R1R2_C1C1_kf', 0.0001),
        Parameter('bind_R1R2_C1C1_kr', 0.1),
        Rule('bind_R1R2_C2', R1(x=None) % R2() + C2(y=None) | R1(x=50) % R2() % C2(y=50), bind_R1R2_C2_kf, bind_R1R2_C2_kr),
        Parameter('bind_R1R2_C2_kf', 0.0002),
        Parameter('bind_R1R2_C2_kr', 0.2),
        Rule('bind_C1C1_R2', C1(y=None, c2=1) % C1(c2=1) + R2(x=None) | C1(y=50, c2=1) % C1(c2=1) % R2(x=50), bind_C1C1_R2_kf, bind_C1C1_R2_kr),
        Parameter('bind_C1C1_R2_kf', 0.0003),
        Parameter('bind_C1C1_R2_kr', 0.3),
         ])

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
                components |= bind_complex(s_row, row_site, s_col, col_site, klist, m1, m2)
    return components

# Catalysis
# =========

def catalyze(enzyme, e_site, substrate, s_site, product, klist):
    """
    Generate the two-step catalytic reaction E + S | E:S >> E + P.

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
        catalyze(E(), 'b', S(), 'b', P(), (1e-4, 1e-1, 1))

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('E', ['b'])
        Monomer('E', ['b'])
        >>> Monomer('S', ['b'])
        Monomer('S', ['b'])
        >>> Monomer('P')
        Monomer('P')
        >>> catalyze(E(), 'b', S(), 'b', P(), (1e-4, 1e-1, 1)) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('bind_E_S_to_ES', E(b=None) + S(b=None) | E(b=1) % S(b=1),
             bind_E_S_to_ES_kf, bind_E_S_to_ES_kr),
         Parameter('bind_E_S_to_ES_kf', 0.0001),
         Parameter('bind_E_S_to_ES_kr', 0.1),
         Rule('catalyze_ES_to_E_P', E(b=1) % S(b=1) >> E(b=None) + P(),
             catalyze_ES_to_E_P_kc),
         Parameter('catalyze_ES_to_E_P_kc', 1.0),
         ])

    Using a single Monomer for substrate and product with a state change::

        Monomer('Kinase', ['b'])
        Monomer('Substrate', ['b', 'y'], {'y': ('U', 'P')})
        catalyze(Kinase(), 'b', Substrate(y='U'), 'b', Substrate(y='P'),
                 (1e-4, 1e-1, 1))

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('Kinase', ['b'])
        Monomer('Kinase', ['b'])
        >>> Monomer('Substrate', ['b', 'y'], {'y': ('U', 'P')})
        Monomer('Substrate', ['b', 'y'], {'y': ('U', 'P')})
        >>> catalyze(Kinase(), 'b', Substrate(y='U'), 'b', Substrate(y='P'), (1e-4, 1e-1, 1)) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('bind_Kinase_SubstrateU_to_KinaseSubstrateU',
             Kinase(b=None) + Substrate(b=None, y='U') | Kinase(b=1) % Substrate(b=1, y='U'),
             bind_Kinase_SubstrateU_to_KinaseSubstrateU_kf,
             bind_Kinase_SubstrateU_to_KinaseSubstrateU_kr),
         Parameter('bind_Kinase_SubstrateU_to_KinaseSubstrateU_kf', 0.0001),
         Parameter('bind_Kinase_SubstrateU_to_KinaseSubstrateU_kr', 0.1),
         Rule('catalyze_KinaseSubstrateU_to_Kinase_SubstrateP',
              Kinase(b=1) % Substrate(b=1, y='U') >> Kinase(b=None) + Substrate(b=None, y='P'),
              catalyze_KinaseSubstrateU_to_Kinase_SubstrateP_kc),
         Parameter('catalyze_KinaseSubstrateU_to_Kinase_SubstrateP_kc', 1.0),
         ])

    """

    _verify_sites(enzyme, e_site)
    _verify_sites(substrate, s_site)

    # Set up some aliases to the patterns we'll use in the rules
    enzyme_free = enzyme({e_site: None})
    # retain any existing state for substrate's s_site, otherwise set it to None
    if s_site in substrate.site_conditions:
        substrate_free = substrate()
        s_state = (substrate.site_conditions[s_site], 1)
    else:
        substrate_free = substrate({s_site: None})
        s_state = 1
    es_complex = enzyme({e_site: 1}) % substrate({s_site: s_state})

    # If product is actually a variant of substrate, we need to explicitly say
    # that it is no longer bound to enzyme, unless product already specifies a
    # state for s_site.
    if product().monomer is substrate().monomer \
            and s_site not in product.site_conditions:
        product = product({s_site: None})

    # create the rules
    components = _macro_rule('bind',
                             enzyme_free + substrate_free | es_complex,
                             klist[0:2], ['kf', 'kr'])
    components |= _macro_rule('catalyze',
                              es_complex >> enzyme_free + product,
                              [klist[2]], ['kc'])

    return components

def catalyze_complex(enzyme, e_site, substrate, s_site, product, klist, m1=None, m2=None):
    """ Generate the two-step catalytic reaction E + S | E:S >> E + P, while allowing complexes to serve as enzyme, substrate and/or product.
        
        E:S1 + S:S2 | E:S1:S:S2 >> E:S1 + P:S2
        
        Parameters
        ----------
        enzyme, substrate, product : Monomer, MonomerPattern, or ComplexPattern
        Monomers or complexes participating in the binding reaction.
        
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
        
        m1, m2 : Monomer or MonomerPattern
        If enzyme or substrate binding site is present in multiple monomers
        within a complex, the specific monomer desired for binding must be specified.
        
        Returns
        -------
        components : ComponentSet
        The generated components. Contains the bidirectional binding Rule
        and optionally three Parameters if klist was given as numbers.
        """
    if isinstance(m1, Monomer):
        m1 = m1()
    if isinstance(m2, Monomer):
        m2 = m2()
        
    def build_complex(s1, site1, m1):
        _verify_sites_complex(s1, site1)
        #Retrieve a dictionary specifying the MonomerPattern within the complex that contains the given binding site.
        specsitesdict = _verify_sites_complex(s1, site1)
        s1complexpatub, s1complexpatb = check_sites_comp_build(s1, site1, m1, specsitesdict)
        return s1complexpatb, s1complexpatub

    def check_sites_comp_build(s1, site1, m1, specsitesdict):
        #Return error if binding site exists on multiple monomers and a monomer for binding (m1) hasn't been specified.
        if len(specsitesdict) > 1 and m1==None:
            raise ValueError("Binding site '%s' present in more than one monomer in complex '%s'.  Specify variable m1, the monomer used for binding within the complex." % (site1, s1))
        if not s1.is_concrete:
            raise ValueError("Complex '%s' must be concrete." % (s1))
            #If the given binding site is only present in one monomer in the complex:
        if m1==None:
            #Build up ComplexPattern for use in rule (with state of given binding site specified).
            s1complexpatub = list(specsitesdict.keys())[0]({site1:None})
            s1complexpatb = list(specsitesdict.keys())[0]({site1:50})
            for monomer in s1.monomer_patterns:
                if monomer not in specsitesdict.keys():
                    s1complexpatub %= monomer
                    s1complexpatb %= monomer
    
        #If the binding site is present on more than one monomer in the complex, the monomer must be specified by the user.  Use specified m1 to build ComplexPattern.
        else:
            #Make sure binding states of MonomerPattern m1 match those of the monomer within the ComplexPattern s1 (ComplexPattern monomer takes precedence if not).
            i = 0
            identical_monomers = []
            for mon in s1.monomer_patterns:
                #Only change the binding site for the first monomer that matches.  Keep any others unchanged to add to final complex that is returned.
                if mon.monomer.name == m1.monomer.name:
                    i += 1
                    if i == 1:
                        s1complexpatub = m1({site1:None})
                        s1complexpatb = m1({site1:50})
                    else:
                        identical_monomers.append(mon)
            #Build up ComplexPattern for use in rule (with state of given binding site  on m1 specified).
            for mon in s1.monomer_patterns:
                if mon.monomer.name != m1.monomer.name:
                    s1complexpatub %= mon
                    s1complexpatb %= mon
            if identical_monomers:
                for i in range(len(identical_monomers)):
                    s1complexpatub %= identical_monomers[i]
                    s1complexpatb %= identical_monomers[i]
    
        return s1complexpatub, s1complexpatb

    #If no complexes exist in the reaction, revert to catalyze().
    if (isinstance(enzyme, MonomerPattern) or isinstance(enzyme, Monomer)) and (isinstance(substrate, MonomerPattern) or isinstance(substrate, Monomer)):
        _verify_sites(enzyme, e_site)
        _verify_sites(substrate, s_site)
        return catalyze(enzyme, e_site, substrate, s_site, product, klist,)
    
    # Build E:S
    if isinstance(enzyme, ComplexPattern):
        enzymepatb, enzyme_free = build_complex(enzyme, e_site, m1)
    else:
        enzymepatb, enzyme_free = enzyme({e_site: 1}), enzyme({e_site: None})
            
    if isinstance(substrate, ComplexPattern):
        substratepatb, substratepatub = build_complex(substrate, s_site, m2)
    else:
        substratepatb = substrate({s_site: 50})
        
        """if s_site in substrate.site_conditions:
            substrate_free = substrate()
            s_state = (substrate.site_conditions[s_site], 1)
        else:
            substrate_free = substrate({s_site: None})
            s_state = 1
        substratepatb = substrate({s_site: s_state})
        """

            
    es_complex = enzymepatb % substratepatb
            
    # Use bind complex to binding rule.

    components = bind_complex(enzyme, e_site, substrate, s_site, klist[0:2], m1, m2)
    components |= _macro_rule('catalyze',
                              es_complex >> enzyme_free + product,
                              [klist[2]], ['kc'])
    return components

def catalyze_state(enzyme, e_site, substrate, s_site, mod_site,
                   state1, state2, klist):
    """
    Generate the two-step catalytic reaction E + S | E:S >> E + P. A wrapper
    around catalyze() with a signature specifying the state change of the
    substrate resulting from catalysis.

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

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('Kinase', ['b'])
        Monomer('Kinase', ['b'])
        >>> Monomer('Substrate', ['b', 'y'], {'y': ('U', 'P')})
        Monomer('Substrate', ['b', 'y'], {'y': ('U', 'P')})
        >>> catalyze_state(Kinase, 'b', Substrate, 'b', 'y', 'U', 'P', (1e-4, 1e-1, 1)) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('bind_Kinase_SubstrateU_to_KinaseSubstrateU',
             Kinase(b=None) + Substrate(b=None, y='U') | Kinase(b=1) % Substrate(b=1, y='U'),
             bind_Kinase_SubstrateU_to_KinaseSubstrateU_kf,
             bind_Kinase_SubstrateU_to_KinaseSubstrateU_kr),
         Parameter('bind_Kinase_SubstrateU_to_KinaseSubstrateU_kf', 0.0001),
         Parameter('bind_Kinase_SubstrateU_to_KinaseSubstrateU_kr', 0.1),
         Rule('catalyze_KinaseSubstrateU_to_Kinase_SubstrateP',
             Kinase(b=1) % Substrate(b=1, y='U') >> Kinase(b=None) + Substrate(b=None, y='P'),
             catalyze_KinaseSubstrateU_to_Kinase_SubstrateP_kc),
         Parameter('catalyze_KinaseSubstrateU_to_Kinase_SubstrateP_kc', 1.0),
         ])

    """

    return catalyze(enzyme, e_site, substrate({mod_site: state1}),
                    s_site, substrate({mod_site: state2}), klist)

def catalyze_one_step(enzyme, substrate, product, kf):
    """
    Generate the one-step catalytic reaction E + S >> E + P.

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
    Convert S to P by E::

        Model()
        Monomer('E', ['b'])
        Monomer('S', ['b'])
        Monomer('P')
        catalyze_one_step(E, S, P, 1e-4)

    If the ability of the enzyme E to catalyze this reaction is dependent
    on the site 'b' of E being unbound, then this macro must be called as

        catalyze_one_step(E(b=None), S, P, 1e-4)

    and similarly if the substrate or product must be unbound.

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('E', ['b'])
        Monomer('E', ['b'])
        >>> Monomer('S', ['b'])
        Monomer('S', ['b'])
        >>> Monomer('P')
        Monomer('P')
        >>> catalyze_one_step(E, S, P, 1e-4) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('one_step_E_S_to_E_P', E() + S() >> E() + P(), one_step_E_S_to_E_P_kf),
         Parameter('one_step_E_S_to_E_P_kf', 0.0001),
         ])

    """

    if isinstance(enzyme, Monomer):
        enzyme = enzyme()
    if isinstance(substrate, Monomer):
        substrate = substrate()
    if isinstance(product, Monomer):
        product = product()
    return _macro_rule('one_step',
                       enzyme + substrate >> enzyme + product,
                       [kf], ['kf'])

def catalyze_one_step_reversible(enzyme, substrate, product, klist):
    """
    Create fwd and reverse rules for catalysis of the form::

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
    One-step, pseudo-first order conversion of S to P by E::

        Model()
        Monomer('E', ['b'])
        Monomer('S', ['b'])
        Monomer('P')
        catalyze_one_step_reversible(E, S, P, [1e-1, 1e-4])

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('E', ['b'])
        Monomer('E', ['b'])
        >>> Monomer('S', ['b'])
        Monomer('S', ['b'])
        >>> Monomer('P')
        Monomer('P')
        >>> catalyze_one_step_reversible(E, S, P, [1e-1, 1e-4]) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('one_step_E_S_to_E_P', E() + S() >> E() + P(), one_step_E_S_to_E_P_kf),
         Parameter('one_step_E_S_to_E_P_kf', 0.1),
         Rule('reverse_P_to_S', P() >> S(), reverse_P_to_S_kr),
         Parameter('reverse_P_to_S_kr', 0.0001),
         ])

    """
    
    if isinstance(enzyme, Monomer):
        enzyme = enzyme()
    if isinstance(substrate, Monomer):
        substrate = substrate()
    if isinstance(product, Monomer):
        product = product()

    components = catalyze_one_step(enzyme, substrate, product, klist[0])

    components |= _macro_rule('reverse', product >> substrate,
                              [klist[1]], ['kr'])
    return components

# Synthesis and degradation
# =========================

def synthesize(species, ksynth):
    """
    Generate a reaction which synthesizes a species.

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
    Synthesize A with site x unbound and site y in state 'e'::

        Model()
        Monomer('A', ['x', 'y'], {'y': ['e', 'f']})
        synthesize(A(x=None, y='e'), 1e-4)

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('A', ['x', 'y'], {'y': ['e', 'f']})
        Monomer('A', ['x', 'y'], {'y': ['e', 'f']})
        >>> synthesize(A(x=None, y='e'), 1e-4) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('synthesize_Ae', None >> A(x=None, y='e'), synthesize_Ae_k),
         Parameter('synthesize_Ae_k', 0.0001),
         ])

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
    """
    Generate a reaction which degrades a species.

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
    Degrade all B, even bound species::

        Model()
        Monomer('B', ['x'])
        degrade(B(), 1e-6)

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('B', ['x'])
        Monomer('B', ['x'])
        >>> degrade(B(), 1e-6) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('degrade_B', B() >> None, degrade_B_k),
         Parameter('degrade_B_k', 1e-06),
         ])

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
    """
    Generate a table of synthesis and degradation reactions.

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
    Specify synthesis and degradation reactions for A and B in a table::

        Model()
        Monomer('A', ['x', 'y'], {'y': ['e', 'f']})
        Monomer('B', ['x'])
        synthesize_degrade_table([[A(x=None, y='e'), 1e-4, 1e-6],
                                  [B(),              None, 1e-7]])

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('A', ['x', 'y'], {'y': ['e', 'f']})
        Monomer('A', ['x', 'y'], {'y': ['e', 'f']})
        >>> Monomer('B', ['x'])
        Monomer('B', ['x'])
        >>> synthesize_degrade_table([[A(x=None, y='e'), 1e-4, 1e-6],
        ...                           [B(),              None, 1e-7]]) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
            Rule('synthesize_Ae', None >> A(x=None, y='e'), synthesize_Ae_k),
            Parameter('synthesize_Ae_k', 0.0001),
            Rule('degrade_Ae', A(x=None, y='e') >> None, degrade_Ae_k),
            Parameter('degrade_Ae_k', 1e-06),
            Rule('degrade_B', B() >> None, degrade_B_k),
            Parameter('degrade_B_k', 1e-07),
            ])

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

# Polymer assembly (pores/rings and chains)
# =========================================

def polymer_species(subunit, site1, site2, size, closed=False):
    """
    Return a ComplexPattern representing a linear or closed circular polymer.

    Parameters
    ----------
    subunit : Monomer or MonomerPattern
        The subunit of which the polymer is composed.
    site1, site2 : string
        The names of the sites where one copy of `subunit` binds to the next.
    size : integer
        The number of subunits in the polymer.
    closed : boolean
        If False (default), the polymer is linear, with unbound sites at each
        end. If True, the polymer is a closed circle, like a ring or pore.

    Returns
    -------
    A ComplexPattern corresponding to the polymer.

    Notes
    -----
    Used by both chain_species and pore_species.

    """
    _verify_sites(subunit, site1, site2)
    if size <= 0:
        raise ValueError("size must be an integer greater than 0")
    if size == 1:
        polymer = subunit({site1: None, site2: None})
    elif size == 2:
        polymer = subunit({site1: None, site2: 1}) % \
                  subunit({site1: 1, site2: None})
    else:
        # If a closed circle, use 0 as the bond number for the "seam";
        # if linear, use None for the unbound ends
        seam_site_num = size if closed else None
        # First subunit
        polymer = subunit({site1: seam_site_num, site2: 1})
        # Build up the ComplexPattern for the polymer, starting with the first
        # subunit
        for i in range(1, size-1):
            polymer %= subunit({site1: i, site2: i + 1})
        # Attach the last subunit
        polymer %= subunit({site1: size-1, site2: seam_site_num})
        # Set ComplexPattern to MatchOnce
        polymer.match_once = True
    return polymer

def assemble_polymer_sequential(subunit, site1, site2, max_size, ktable,
                                closed=False):
    """Generate rules to assemble a polymer by sequential subunit addition.

    The polymer species are created by sequential addition of `subunit` monomers,
    i.e. larger oligomeric species never fuse together. The polymer structure is
    defined by the `polymer_species` macro.

    Parameters
    ----------
    subunit : Monomer or MonomerPattern
        The subunit of which the polymer is composed.
    site1, site2 : string
        The names of the sites where one copy of `subunit` binds to the next.
    max_size : integer
        The maximum number of subunits in the polymer.
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
        generated names based on `subunit`, `site1`, `site2` and the polymer sizes
        and these parameters will be included at the end of the returned
        component list.
    closed : boolean
        If False (default), assembles a linear (non-circular) polymer. If True,
        assembles a circular ring/pore polymer.

    Notes
    -----

    See documentation for :py:func:`assemble_chain_sequential` and
    :py:func:`assemble_pore_sequential` for examples.

    """
    if len(ktable) != max_size - 1:
        raise ValueError("len(ktable) must be equal to max_size - 1")

    def polymer_rule_name(rule_expression, size):
        react_p = rule_expression.reactant_pattern
        monomer = react_p.complex_patterns[0].monomer_patterns[0].monomer
        return '%s_%d' % (monomer.name, size)

    components = ComponentSet()
    s = polymer_species(subunit, site1, site2, 1, closed=closed)
    for size, klist in zip(range(2, max_size + 1), ktable):
        polymer_prev = polymer_species(subunit, site1, site2, size - 1,
                                       closed=closed)
        polymer_next = polymer_species(subunit, site1, site2, size,
                                       closed=closed)
        name_func = functools.partial(polymer_rule_name, size=size)
        rule_name_base = 'assemble_%s_sequential' % \
                         ('pore' if closed else 'chain')
        components |= _macro_rule(rule_name_base,
                                  s + polymer_prev | polymer_next,
                                  klist, ['kf', 'kr'],
                                  name_func=name_func)
    return components

# Pore assembly
# =============

def pore_species(subunit, site1, site2, size):
    """
    Return a ComplexPattern representing a circular homomeric pore.

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
    A ComplexPattern corresponding to the pore.

    Notes
    -----
    At sizes 1 and 2 the ring is not closed, i.e. there is one site1 and one
    site2 which remain unbound. At size 3 and up the ring is closed and all
    site1 sites are bound to a site2.

    Examples
    --------
    Get the ComplexPattern object representing a pore of size 4::

        Model()
        Monomer('Unit', ['p1', 'p2'])
        pore_tetramer = pore_species(Unit, 'p1', 'p2', 4)

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('Unit', ['p1', 'p2'])
        Monomer('Unit', ['p1', 'p2'])
        >>> pore_species(Unit, 'p1', 'p2', 4)
        MatchOnce(Unit(p1=4, p2=1) % Unit(p1=1, p2=2) % Unit(p1=2, p2=3) % Unit(p1=3, p2=4))

    """
    return polymer_species(subunit, site1, site2, size, closed=True)

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

    Examples
    --------
    Assemble a three-membered pore by sequential addition of monomers,
    with the same forward/reverse rates for monomer-monomer and monomer-dimer
    interactions::

        Model()
        Monomer('Unit', ['p1', 'p2'])
        assemble_pore_sequential(Unit, 'p1', 'p2', 3, [[1e-4, 1e-1]] * 2)

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('Unit', ['p1', 'p2'])
        Monomer('Unit', ['p1', 'p2'])
        >>> assemble_pore_sequential(Unit, 'p1', 'p2', 3, [[1e-4, 1e-1]] * 2) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('assemble_pore_sequential_Unit_2',
              Unit(p1=None, p2=None) + Unit(p1=None, p2=None) |
                  Unit(p1=None, p2=1) % Unit(p1=1, p2=None),
              assemble_pore_sequential_Unit_2_kf,
              assemble_pore_sequential_Unit_2_kr),
         Parameter('assemble_pore_sequential_Unit_2_kf', 0.0001),
         Parameter('assemble_pore_sequential_Unit_2_kr', 0.1),
         Rule('assemble_pore_sequential_Unit_3',
              Unit(p1=None, p2=None) + Unit(p1=None, p2=1) % Unit(p1=1, p2=None) |
                  MatchOnce(Unit(p1=3, p2=1) % Unit(p1=1, p2=2) % Unit(p1=2, p2=3)),
              assemble_pore_sequential_Unit_3_kf,
              assemble_pore_sequential_Unit_3_kr),
         Parameter('assemble_pore_sequential_Unit_3_kf', 0.0001),
         Parameter('assemble_pore_sequential_Unit_3_kr', 0.1),
         ])

    """
    return assemble_polymer_sequential(subunit, site1, site2, max_size, ktable,
                                       closed=True)

def pore_transport(subunit, sp_site1, sp_site2, sc_site, min_size, max_size,
                   csource, c_site, cdest, ktable):
    """
    Generate rules to transport cargo through a circular homomeric pore.

    The pore structure is defined by the `pore_species` macro -- `subunit`
    monomers bind to each other from `sp_site1` to `sp_site2` to form a closed
    ring. The transport reaction is modeled as a catalytic process of the form
    pore + csource | pore:csource >> pore + cdest

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

    Examples
    --------
    Specify that a three-membered pore is capable of
    transporting cargo from the mitochondria to the cytoplasm::

        Model()
        Monomer('Unit', ['p1', 'p2', 'sc_site'])
        Monomer('Cargo', ['c_site', 'loc'], {'loc':['mito', 'cyto']})
        pore_transport(Unit, 'p1', 'p2', 'sc_site', 3, 3,
                       Cargo(loc='mito'), 'c_site', Cargo(loc='cyto'),
                       [[1e-4, 1e-1, 1]])

    Generates two rules--one (reversible) binding rule and one transport
    rule--and the three associated parameters.

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('Unit', ['p1', 'p2', 'sc_site'])
        Monomer('Unit', ['p1', 'p2', 'sc_site'])
        >>> Monomer('Cargo', ['c_site', 'loc'], {'loc':['mito', 'cyto']})
        Monomer('Cargo', ['c_site', 'loc'], {'loc': ['mito', 'cyto']})
        >>> pore_transport(Unit, 'p1', 'p2', 'sc_site', 3, 3,
        ...                Cargo(loc='mito'), 'c_site', Cargo(loc='cyto'),
        ...                [[1e-4, 1e-1, 1]]) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('pore_transport_complex_Unit_3_Cargomito',
             MatchOnce(Unit(p1=3, p2=1, sc_site=None) %
                 Unit(p1=1, p2=2, sc_site=None) %
                 Unit(p1=2, p2=3, sc_site=None)) +
                 Cargo(c_site=None, loc='mito') |
             MatchOnce(Unit(p1=3, p2=1, sc_site=4) %
                 Unit(p1=1, p2=2, sc_site=None) %
                 Unit(p1=2, p2=3, sc_site=None) %
                 Cargo(c_site=4, loc='mito')),
             pore_transport_complex_Unit_3_Cargomito_kf,
             pore_transport_complex_Unit_3_Cargomito_kr),
         Parameter('pore_transport_complex_Unit_3_Cargomito_kf', 0.0001),
         Parameter('pore_transport_complex_Unit_3_Cargomito_kr', 0.1),
         Rule('pore_transport_dissociate_Unit_3_Cargocyto',
             MatchOnce(Unit(p1=3, p2=1, sc_site=4) %
                 Unit(p1=1, p2=2, sc_site=None) %
                 Unit(p1=2, p2=3, sc_site=None) %
                 Cargo(c_site=4, loc='mito')) >>
             MatchOnce(Unit(p1=3, p2=1, sc_site=None) %
                 Unit(p1=1, p2=2, sc_site=None) %
                 Unit(p1=2, p2=3, sc_site=None)) +
                 Cargo(c_site=None, loc='cyto'),
             pore_transport_dissociate_Unit_3_Cargocyto_kc),
         Parameter('pore_transport_dissociate_Unit_3_Cargocyto_kc', 1.0),
         ])

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
                                  pore_free + csource_free | pc_complex,
                                  klist[0:2], ['kf', 'kr'],
                                  name_func=name_func)
        components |= _macro_rule('pore_transport_dissociate',
                                  pc_complex >> pore_free + cdest,
                                  [klist[2]], ['kc'],
                                  name_func=name_func)

    return components

def pore_bind(subunit, sp_site1, sp_site2, sc_site, size, cargo, c_site,
              klist):
    """
    Generate rules to bind a monomer to a circular homomeric pore.

    The pore structure is defined by the `pore_species` macro -- `subunit`
    monomers bind to each other from `sp_site1` to `sp_site2` to form a closed
    ring. The binding reaction takes the form pore + cargo | pore:cargo.

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

    Examples
    --------
    Specify that a cargo molecule can bind reversibly to a 3-membered
    pore::

        Model()
        Monomer('Unit', ['p1', 'p2', 'sc_site'])
        Monomer('Cargo', ['c_site'])
        pore_bind(Unit, 'p1', 'p2', 'sc_site', 3, 
                  Cargo(), 'c_site', [1e-4, 1e-1, 1])

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('Unit', ['p1', 'p2', 'sc_site'])
        Monomer('Unit', ['p1', 'p2', 'sc_site'])
        >>> Monomer('Cargo', ['c_site'])
        Monomer('Cargo', ['c_site'])
        >>> pore_bind(Unit, 'p1', 'p2', 'sc_site', 3, 
        ...           Cargo(), 'c_site', [1e-4, 1e-1, 1]) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('pore_bind_Unit_3_Cargo',
             MatchOnce(Unit(p1=3, p2=1, sc_site=None) %
                 Unit(p1=1, p2=2, sc_site=None) %
                 Unit(p1=2, p2=3, sc_site=None)) +
                 Cargo(c_site=None) |
             MatchOnce(Unit(p1=3, p2=1, sc_site=4) %
                 Unit(p1=1, p2=2, sc_site=None) %
                 Unit(p1=2, p2=3, sc_site=None) %
                 Cargo(c_site=4)),
             pore_bind_Unit_3_Cargo_kf, pore_bind_Unit_3_Cargo_kr),
         Parameter('pore_bind_Unit_3_Cargo_kf', 0.0001),
         Parameter('pore_bind_Unit_3_Cargo_kr', 0.1),
         ])

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
                              pore_free + cargo_free | pc_complex,
                              klist[0:2], ['kf', 'kr'],
                              name_func=name_func)

    return components

# Chain assembly
# =============

def chain_species(subunit, site1, site2, size):
    """
    Return a ComplexPattern representing a linear, chained polymer.

    Parameters
    ----------
    subunit : Monomer or MonomerPattern
        The subunit of which the chain is composed.
    site1, site2 : string
        The names of the sites where one copy of `subunit` binds to the next.
    size : integer
        The number of subunits in the chain.

    Returns
    -------
    A ComplexPattern corresponding to the chain.

    Notes
    -----
    Similar to pore_species, but never closes the chain.

    Examples
    --------
    Get the ComplexPattern object representing a chain of length 4::

        Model()
        Monomer('Unit', ['p1', 'p2'])
        chain_tetramer = chain_species(Unit, 'p1', 'p2', 4)

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('Unit', ['p1', 'p2'])
        Monomer('Unit', ['p1', 'p2'])
        >>> chain_species(Unit, 'p1', 'p2', 4)
        MatchOnce(Unit(p1=None, p2=1) % Unit(p1=1, p2=2) % Unit(p1=2, p2=3) % Unit(p1=3, p2=None))

    """
    return polymer_species(subunit, site1, site2, size, closed=False)

def assemble_chain_sequential(subunit, site1, site2, max_size, ktable):
    """
    Generate rules to assemble a homomeric chain sequentially.

    The chain species are created by sequential addition of `subunit` monomers.
    The chain structure is defined by the `chain_species` macro.

    Parameters
    ----------
    subunit : Monomer or MonomerPattern
        The subunit of which the chain is composed.
    site1, site2 : string
        The names of the sites where one copy of `subunit` binds to the next.
    max_size : integer
        The maximum number of subunits in the chain.
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
        generated names based on `subunit`, `site1`, `site2` and the chain sizes
        and these parameters will be included at the end of the returned
        component list.

    Examples
    --------
    Assemble a three-membered chain by sequential addition of monomers,
    with the same forward/reverse rates for monomer-monomer and monomer-dimer
    interactions::

        Model()
        Monomer('Unit', ['p1', 'p2'])
        assemble_chain_sequential(Unit, 'p1', 'p2', 3, [[1e-4, 1e-1]] * 2)

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('Unit', ['p1', 'p2'])
        Monomer('Unit', ['p1', 'p2'])
        >>> assemble_chain_sequential(Unit, 'p1', 'p2', 3, [[1e-4, 1e-1]] * 2) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('assemble_chain_sequential_Unit_2', Unit(p1=None, p2=None) + Unit(p1=None, p2=None) | Unit(p1=None, p2=1) % Unit(p1=1, p2=None), assemble_chain_sequential_Unit_2_kf, assemble_chain_sequential_Unit_2_kr),
         Parameter('assemble_chain_sequential_Unit_2_kf', 0.0001),
         Parameter('assemble_chain_sequential_Unit_2_kr', 0.1),
         Rule('assemble_chain_sequential_Unit_3', Unit(p1=None, p2=None) + Unit(p1=None, p2=1) % Unit(p1=1, p2=None) | MatchOnce(Unit(p1=None, p2=1) % Unit(p1=1, p2=2) % Unit(p1=2, p2=None)), assemble_chain_sequential_Unit_3_kf, assemble_chain_sequential_Unit_3_kr),
         Parameter('assemble_chain_sequential_Unit_3_kf', 0.0001),
         Parameter('assemble_chain_sequential_Unit_3_kr', 0.1),
         ])

    """
    return assemble_polymer_sequential(subunit, site1, site2, max_size, ktable,
                                       closed=False)

def chain_species_base(base, basesite, subunit, site1, site2, size, comp=1):
    """
    Return a MonomerPattern representing a chained species, chained to a base complex.

    Parameters
    ----------
    base : Monomer or MonomerPattern
        The base complex to which the growing chain will be attached.
    basesite : string
        Name of the site on complex where first subunit binds.
    subunit : Monomer or MonomerPattern
        The subunit of which the chain is composed.
    site1, site2 : string
        The names of the sites where one copy of `subunit` binds to the next.
    size : integer
        The number of subunits in the chain.
    comp : optional; a ComplexPattern to which the base molecule is attached.

    Returns
    -------
    A ComplexPattern corresponding to the chain.

    Notes
    -----
    Similar to pore_species, but never closes the chain.

    Examples
    --------
    Get the ComplexPattern object representing a chain of size 4 bound to a base, which is itself bound to a complex:

        Model()
        Monomer('Base', ['b1', 'b2'])
        Monomer('Unit', ['p1', 'p2'])
        Monomer('Complex1', ['s1'])
        Monomer('Complex2', ['s1', 's2'])
        chain_tetramer = chain_species_base(Base(b1=1, b2=ANY), 'b1', Unit, 'p1', 'p2', 4, Complex1(s1=ANY) % Complex2(s1=ANY, s2=ANY))

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('Unit', ['p1', 'p2'])
        Monomer('Unit', ['p1', 'p2'])
        >>> Monomer('Base', ['b1', 'b2'])
        Monomer('Base', ['b1', 'b2'])
        >>> Monomer('Complex1', ['s1'])
        Monomer('Complex1', ['s1'])
        >>> Monomer('Complex2', ['s1', 's2'])
        Monomer('Complex2', ['s1', 's2'])
        >>> chain_species_base(Base(b2=ANY), 'b1', Unit, 'p1', 'p2', 4, Complex1(s1=ANY) % Complex2(s1=ANY, s2=ANY))
        MatchOnce(Complex1(s1=ANY) % Complex2(s1=ANY, s2=ANY) % Base(b1=1, b2=ANY) % Unit(p1=1, p2=2) % Unit(p1=2, p2=3) % Unit(p1=3, p2=4) % Unit(p1=4, p2=None))
    """
    _verify_sites(base, basesite)
    _verify_sites(subunit, site1, site2)
    if size <= 0:
        raise ValueError("size must be an integer greater than 0")
    if comp == 1:
        compbase = base({basesite: 1})
    else:
        compbase = comp % base({basesite: 1})
    if size == 1:
        chainlink = compbase % subunit({site1: 1, site2: None})
    elif size == 2:
        chainlink = compbase % subunit({site1: 1, site2: 2}) % \
            subunit({site1: 2, site2: None})
    else:
      # build up a ComplexPattern, starting with a single subunit
        chainbase = compbase
        chainlink = chainbase % subunit({site1: 1, site2: 2})
        for i in range(2, size):
            chainlink %= subunit({site1: i, site2: i+1})
        chainlink %= subunit({site1: size, site2: None})
        chainlink.match_once = True  
    
    return chainlink

def assemble_chain_sequential_base(base, basesite, subunit, site1, site2, max_size, ktable, comp=1):
    """
    Generate rules to assemble a homomeric chain sequentially onto a base complex (only the subunit creates repeating chain, not the base).

    The chain species are created by sequential addition of `subunit` monomers.
    The chain structure is defined by the `pore_species_base` macro.

    Parameters
    ----------
    base : Monomer or MonomerPattern
        The base complex to which the chain is attached.
    basesite : string
        The name of the site on the complex to which chain attaches.
    subunit : Monomer or MonomerPattern
        The subunit of which the chain is composed.
    site1, site2 : string
        The names of the sites where one copy of `subunit` binds to the next; the first will also be the site where the first subunit binds the base.
    max_size : integer
        The maximum number of subunits in the chain.
    ktable : list of lists of Parameters or numbers
        Table of forward and reverse rate constants for the assembly steps. The
        outer list must be of length `max_size` + 1, and the inner lists must
        all be of length 2. In the outer list, the first element corresponds to
        the first assembly step in which the complex binds the first subunit.  The next corresponds to a bound subunit binding to form a
        2-subunit complex, and the last element corresponds to the final step in
        which the `max_size`th subunit is added. Each inner list contains the
        forward and reverse rate constants (in that order) for the corresponding
        assembly reaction, and each of these pairs must comprise solely
        Parameter objects or solely numbers (never one of each). If Parameters
        are passed, they will be used directly in the generated Rules. If
        numbers are passed, Parameters will be created with automatically
        generated names based on `subunit`, `site1`, `site2` and the chain sizes
        and these parameters will be included at the end of the returned
        component list.
    comp : optional; a ComplexPattern to which the base molecule is attached.

    Examples
    --------
    Assemble a three-membered chain by sequential addition of monomers to a base, which is in turn attached to a complex,
    with the same forward/reverse rates for monomer-monomer and monomer-dimer
    interactions::

        Model()
        Monomer('Base', ['b1', 'b2'])
        Monomer('Unit', ['p1', 'p2'])
        Monomer('Complex1', ['s1'])
        Monomer('Complex2', ['s1', s2'])
        assemble_chain_sequential(Base(b2=ANY), 'b1', Unit, 'p1', 'p2', 3, [[1e-4, 1e-1]] * 2, Complex1(s1=ANY) % Complex2(s1=ANY, s2=ANY))

    Execution::

        >>> Model() # doctest:+ELLIPSIS
        <Model '_interactive_' (monomers: 0, rules: 0, parameters: 0, expressions: 0, compartments: 0) at ...>
        >>> Monomer('Base', ['b1', 'b2'])
        Monomer('Base', ['b1', 'b2'])
        >>> Monomer('Unit', ['p1', 'p2'])
        Monomer('Unit', ['p1', 'p2'])
        >>> Monomer('Complex1', ['s1'])
        Monomer('Complex1', ['s1'])
        >>> Monomer('Complex2', ['s1', 's2'])
        Monomer('Complex2', ['s1', 's2'])
        >>> assemble_chain_sequential_base(Base(b2=ANY), 'b1', Unit, 'p1', 'p2', 3, [[1e-4, 1e-1]] * 2, Complex1(s1=ANY) % Complex2(s1=ANY, s2=ANY)) # doctest:+NORMALIZE_WHITESPACE
        ComponentSet([
         Rule('assemble_chain_sequential_base_Unit_2', Unit(p1=None, p2=None) + Complex1(s1=ANY) % Complex2(s1=ANY, s2=ANY) % Base(b1=1, b2=ANY) % Unit(p1=1, p2=None) | Complex1(s1=ANY) % Complex2(s1=ANY, s2=ANY) % Base(b1=1, b2=ANY) % Unit(p1=1, p2=2) % Unit(p1=2, p2=None), assemble_chain_sequential_base_Unit_2_kf, assemble_chain_sequential_base_Unit_2_kr),
         Parameter('assemble_chain_sequential_base_Unit_2_kf', 0.0001),
         Parameter('assemble_chain_sequential_base_Unit_2_kr', 0.1),
         Rule('assemble_chain_sequential_base_Unit_3', Unit(p1=None, p2=None) + Complex1(s1=ANY) % Complex2(s1=ANY, s2=ANY) % Base(b1=1, b2=ANY) % Unit(p1=1, p2=2) % Unit(p1=2, p2=None) | MatchOnce(Complex1(s1=ANY) % Complex2(s1=ANY, s2=ANY) % Base(b1=1, b2=ANY) % Unit(p1=1, p2=2) % Unit(p1=2, p2=3) % Unit(p1=3, p2=None)), assemble_chain_sequential_base_Unit_3_kf, assemble_chain_sequential_base_Unit_3_kr),
         Parameter('assemble_chain_sequential_base_Unit_3_kf', 0.0001),
         Parameter('assemble_chain_sequential_base_Unit_3_kr', 0.1),
         ])

    """

    if len(ktable) != max_size-1:
        raise ValueError("len(ktable) must be equal to max_size-1")

    def chain_rule_name(rule_expression, size):
        react_p = rule_expression.reactant_pattern
        monomer = react_p.complex_patterns[0].monomer_patterns[0].monomer
        return '%s_%d' % (monomer.name, size)

    components = ComponentSet()
    s = subunit({site1:None, site2:None})
    for size, klist in zip(range(2, max_size + 1), ktable):
        chain_prev = chain_species_base(base, basesite, subunit, site1, site2, size - 1, comp)
        chain_next = chain_species_base(base, basesite, subunit, site1, site2, size, comp)
        name_func = functools.partial(chain_rule_name, size=size)
        components |= _macro_rule('assemble_chain_sequential_base',
                                  s + chain_prev | chain_next,
                                  klist, ['kf', 'kr'],
                                  name_func=name_func)

    return components

if __name__ == "__main__":
    import doctest
    doctest.testmod()

