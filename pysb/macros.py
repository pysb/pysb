import inspect
from pysb import *
import pysb.core
from pysb.core import ComponentSet
import numbers

DEFAULT_UNI_KF = 1e6 # In units of sec^-1
DEFAULT_BI_KF = 1e-4 # In units of nM^-1 sec^-1
DEFAULT_KR = 0.1     # In units of sec^-1
DEFAULT_KC = 1       # In units of sec^-1


def complex_pattern_label(cp):
    """Return a string label for a ComplexPattern."""
    mp_labels = [monomer_pattern_label(mp) for mp in cp.monomer_patterns]
    return ''.join(mp_labels)

# TODO: Check to make sure it's a monomer
def monomer_pattern_label(mp):
    """Return a reasonable string label for a MonomerPattern."""
    site_values = [str(x) for x in mp.site_conditions.values() if x is not None]
    return mp.monomer.name + ''.join(site_values)

def _macro_rule(rule_basename, rule_expression, klist):
    """
    TODO: Write comment
    """

    # Get reactant pattern
    react_p = rule_expression.reactant_pattern
    prod_p = rule_expression.product_pattern
    # Build the rule_name
    lhs_label = [complex_pattern_label(cp) for cp in react_p.complex_patterns]
    print lhs_label
    lhs_label = '_'.join(lhs_label)
    print lhs_label
    rhs_label = [complex_pattern_label(cp) for cp in prod_p.complex_patterns]
    print rhs_label
    rhs_label = '_'.join(rhs_label)
    print rhs_label
    r_name = '%s_%s_to_%s' % (rule_basename, lhs_label, rhs_label)

    # If rule is unidirectional, make sure we only have one parameter
    if (not rule_expression.is_reversible):
        if len(klist) != 1:
            raise ValueError("A unidirectional rule must have one parameter.")
    # If rule is bidirectional, make sure we have two parameters
    else:
        if len(klist) != 2:
            raise ValueError("A bidirectional rule must have two parameters.")

    if all(isinstance(x, Parameter) for x in klist):
        k1 = klist[0]
        if rule_expression.is_reversible:
            k2 = klist[1]
        params_created = ComponentSet([])
    # if klist is numbers, generate the Parameters
    elif (isinstance(klist, dict) and
         all(isinstance(x, str) for x in klist.keys()) and
         all(isinstance(x, numbers.Real) for x in klist.values())):
        k1 = Parameter('%s_%s' % (r_name, klist.keys()[0]), klist.values()[0])
        params_created = ComponentSet([k1]) 
        if rule_expression.is_reversible:
            k2 = Parameter('%s_%s' % (r_name, klist.keys()[1]),
                           klist.values()[1])
            params_created |= ComponentSet([k2])
    else:
        raise ValueError("klist must contain Parameter objects or " +
            "a dict of names and numbers")

    if rule_expression.is_reversible:
        r = Rule(r_name, rule_expression, k1, k2)
    else:
        r = Rule(r_name, rule_expression, k1)

    # Build a set of components that were created
    return ComponentSet([r]) | params_created



## Unimolecular patterns
def two_state_equilibrium(s1, s2, klist):
    """Generate unimolecular reversible equilibrium reaction S1 <-> S2.

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
    s1 = s1()
    s2 = s2()

    # generate the rule names
    # FIXME: this will fail if the argument passed is a complex,
    # or a Monomer object... 
    s1_name = monomer_pattern_label(s1)
    s2_name = monomer_pattern_label(s2)
    r_name = 'equilibrate_%s_%s' % (s1_name, s2_name)
    
    if all(isinstance(x, Parameter) for x in klist):
        kf, kr = klist
        params_created = False
    elif all(isinstance(x, numbers.Real) for x in klist):
        # if klist is numbers, generate the Parameters
        kf = Parameter(r_name + '_kf', klist[0])
        kr = Parameter(r_name + '_kr', klist[1])
        params_created = True
    else:
        raise ValueError("klist must contain Parameters objects or numbers")

    # create the rule
    r = Rule(r_name, s1 <> s2, kf, kr)

    # build a set of components that were created
    components = ComponentSet([r])
    if params_created:
        components |= ComponentSet([kf, kr])

    return components

## Binding
#TODO: Finish writing docstring
def bind(s1, site1, s2, site2, klist):
    """Automation of the s1 + s2 <> s1:s2 one-step complex formation,
    but allows the binding sites of both species to be specified. Note that it
    expects s1 and s2 to be MonomerPatterns (not Monomers), and site1
    and site2 to be strings indicating the names of the binding sites.
    klist is list of Parameter objects or numbers.
    """

    # FIXME: this will fail if the argument passed is a complex
    # turn any Monomers into MonomerPatterns
    s1 = s1()
    s2 = s2()

    # verify that sites are valid
    if site1 not in s1.monomer.sites_dict:
        raise ValueError("site1 '%s' not present in s1 '%s'" %
                         (site1, s1.monomer.name))
    if site2 not in s2.monomer.sites_dict:
        raise ValueError("site2 '%s' not present in s2 '%s'" %
                         (site2, s2.monomer.name))

    # generate the rule names
    # FIXME: this will fail if the argument passed is a Complex object. 
    s1_name = monomer_pattern_label(s1)
    s2_name = monomer_pattern_label(s2)
    rc_name = 'bind_%s_%s' % (s1_name, s2_name)

    # set up some aliases to the patterns we'll use in the rules
    enzyme_free = enzyme({e_site: None})
    substrate_free = substrate({s_site: None})
    es_complex = enzyme({e_site: 1}) % substrate({s_site: 1})
    # if product is actually a variant of substrate, we need to explicitly say
    # that it is no longer bound to enzyme
    if product.monomer is substrate.monomer:
        product = product({e_site: None})

    if all(isinstance(x, Parameter) for x in klist):
        kf, kr, kc = klist
        params_created = False
    elif all(isinstance(x, numbers.Real) for x in klist):
        # if klist is numbers, generate the Parameters
        kf = Parameter(rc_name + '_kf', klist[0])
        kr = Parameter(rc_name + '_kr', klist[1])
        kc = Parameter(rd_name + '_kc', klist[2])
        params_created = True
    else:
        raise ValueError("klist must contain Parameters objects or numbers")
     
    # create the rules
    rc = Rule(rc_name, enzyme_free + substrate_free <> es_complex, kf, kr)
    rd = Rule(rd_name, es_complex >> enzyme_free + product, kc)

    # build a set of components that were created
    components = ComponentSet([rc, rd])
    if params_created:
        components |= ComponentSet([kf, kr, kc])

    return components

#TODO: Refactor
def bind_table(bindtable, row_site, col_site):
    """This assumes that the monomers passed are in their desired state without
    the sites which will be used for binding.
    bindtable is a list of lists denoting the reactions between two types of
    reactants as follows:

    bindtable[0]: [                     reactypeA0,  ...,  reactypeAN]
    bindtable[1]: [reactypeB0,  (fwdrate, revrate),  ...,            ]
    bindtable[2]: [reactypeB1,                 ,     ...,            ]

    To indicate that no interaction occurs, simply enter None in the bind table.
    """

    # TODO return created components
    # TODO handle parameter objects and numbers
    # TODO full comments

    # parse the list, extract reactants, products and parameter families
    #first line is one set of reactants
    react_cols = bindtable[0]
    react_rows = [row[0] for row in bindtable[1:]]

    # Notice this makes intrxns indexed by intrxns[row][col]
    intrxns = [row[1:] for row in bindtable[1:]]

    # loop over interactions
    pc = 1 # parameter counter
    rc = 1 # rule counter, easy way of making sure names don't clash #FIXME
    for i in range(0, len(react_rows)):
        for j in range(0, len(react_cols)):
            if intrxns[i][j] is not None:
                kf, kr = intrxns[i][j]
                row_mpattern = react_rows[i]()
                col_mpattern = react_cols[j]()
                kf_parm = Parameter('bt%d%d_kf' % (i, j), kf)
                kr_parm = Parameter('bt%d%d_kr' % (i, j), kr)

                bind(react_rows[i](), row_site, react_cols[j](), col_site,
                     [kf_parm, kr_parm])


## Catalysis
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
    
    # turn any Monomers into MonomerPatterns
    substrate = substrate()
    enzyme = enzyme()
    product = product()

    # verify that sites are valid
    if e_site not in enzyme.monomer.sites_dict:
        raise ValueError("e_site '%s' not present in monomer '%s'" %
                         (e_site, enzyme.monomer.name))
    if s_site not in substrate.monomer.sites_dict:
        raise ValueError("s_site '%s' not present in monomer '%s'" %
                         (s_site, substrate.monomer.name))

    # generate the rule names
    # FIXME: this will fail if the argument passed is a Complex object. 
    substrate_name = monomer_pattern_label(substrate)
    enzyme_name = monomer_pattern_label(enzyme)
    product_name = monomer_pattern_label(product)
    rc_name = 'complex_%s_%s' % (substrate_name, enzyme_name)
    rd_name = 'dissociate_%s_from_%s' % (product_name, enzyme_name)

    # set up some aliases to the patterns we'll use in the rules
    enzyme_free = enzyme({e_site: None})
    substrate_free = substrate({s_site: None})
    es_complex = enzyme({e_site: 1}) % substrate({s_site: 1})
    # if product is actually a variant of substrate, we need to explicitly say
    # that it is no longer bound to enzyme
    if product.monomer is substrate.monomer:
        product = product({e_site: None})

    if all(isinstance(x, Parameter) for x in klist):
        kf, kr, kc = klist
        params_created = False
    elif all(isinstance(x, numbers.Real) for x in klist):
        # if klist is numbers, generate the Parameters
        kf = Parameter(rc_name + '_kf', klist[0])
        kr = Parameter(rc_name + '_kr', klist[1])
        kc = Parameter(rd_name + '_kc', klist[2])
        params_created = True
    else:
        raise ValueError("klist must contain Parameters objects or numbers")
     
    # create the rules
    rc = Rule(rc_name, enzyme_free + substrate_free <> es_complex, kf, kr)
    rd = Rule(rd_name, es_complex >> enzyme_free + product, kc)

    # build a set of components that were created
    components = ComponentSet([rc, rd])
    if params_created:
        components |= ComponentSet([kf, kr, kc])

    return components


"""
catalyze(e, e_site, s, s_site, prod, klist)
    convert all to MPs by apply(label, monomers)
    check s_site on s, e_site on e
    make rulepatterns
    for each rulepattern
        comps += macro_me(basename, rulepattern?)
    return comps 
"""



def catalyze_state(enzyme, e_site, substrate, s_site, mod_site,
                   state1, state2, klist):
    """Generate the two-step catalytic reaction E + S <> E:S >> E + P.
    A wrapper around catalyze() with a signature specifying the state change
    of the substrate that resulting from catalysis.

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

#TODO: Implement
def catalyze_table():
    pass

def catalyze_one_step(enzyme, substrate, product, kf):
    """Automation of the Enz + Sub >> Enz + Prod one-step catalytic reaction.
    Assumes state of Enz is unchanged.

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
    Using distinct Monomers for substrate and product::

        Model()
        Monomer('E', ['b'])
        Monomer('S', ['b'])
        Monomer('P')
        catalyze_one_step(E, S, P, 1e-4)

    If the ability of the enzyme E to catalyze this reaction is dependent
    on the site 'b' of E being unbound, then this macro must be called as

        catalyze_one_step(E(b=None), S, P, 1e-4)
        
    and similarly if the substrate must be unbound.

    Using a single Monomer for substrate and product with a state change::

        Monomer('Kinase', ['b'])
        Monomer('Substrate', ['b', 'y'], {'y': ('U', 'P')})
        catalyze(Kinase, Substrate(y='U'), Substrate(y='P'), 1e-4)

    """
    # Turn any Monomers into MonomerPatterns
    substrate = substrate()
    enzyme = enzyme()
    product = product()

    # Generate the rule names
    # FIXME: this will fail if the argument passed is a Complex object. 
    substrate_name = monomer_pattern_label(substrate)
    enzyme_name = monomer_pattern_label(enzyme)
    product_name = monomer_pattern_label(product)

    r_name = 'one_step_%s_to_%s_by_%s' % (substrate_name, product_name, enzyme_name)

    # 
    if all(isinstance(x, Parameter) for x in klist):
        kf, kr, kc = klist
        params_created = False
    elif all(isinstance(x, numbers.Real) for x in klist):
        # if klist is numbers, generate the Parameters
        kf = Parameter(rc_name + '_kf', klist[0])
        kr = Parameter(rc_name + '_kr', klist[1])
        kc = Parameter(rd_name + '_kc', klist[2])
        params_created = True
    else:
        raise ValueError("klist must contain Parameters objects or numbers")
     
    # if kf is a number, generate the Parameter
    if isinstance(kf, Parameter):
        params_created = False
    elif isinstance(kf, numbers.Real):
        kf = Parameter(r_name + '_kf', kf)
        params_created = True
    else:
        raise ValueError("klist must contain Parameters objects or numbers")

    # Write the rule
    r = Rule(r_name, enzyme + substrate >> enzyme + product, kf)
   
    # build a set of components that were created
    components = ComponentSet([rc, rd])
    if params_created:
        components |= ComponentSet([kf, kr, kc])

    return components
 
    # Return the components created by this function
    components = ComponentSet([r])
    if params_created:
        components |= ComponentSet([kf])

    return components_created

def catalyze_one_step_reversible(sub, enz, prod, klist=None):
    """Create fwd and reverse rules for catalysis of the form:
        A + B -> A + C
            C -> A

    Creates two rules with names following the pattern: 'cat_sub_to_prod'
    and 'prod_to_sub'.

    The function generates a rule with the name following the pattern

    * sub is a MonomerPattern specifying the species that is acted upon.
    * enz is a MonomerPattern specifying the species that determines the rate
      of the reaction.
      NO BINDING OCCURS BETWEEN THE SPECIES.
    * prod is a MonomerPattern specifying the state of the sites of sub after
      catalysis.
      ANY SITES THAT ARE SPECIFIED IN SUB SHOULD BE SPECIFIED FOR PROD AND
      VICE VERSA.
    * klist is a list of Parameter objects specifying the forward (sub to prod)
      and reverse (prod to sub) rates. If not specified, the parameters are
      generated according to the pattern 'cat_sub_to_prod_rate' and
      'prod_to_sub_rate'.
    """

    # FIXME: this will fail if the argument passed is a complex, or a Monomer object... 
    r_name_fwd = 'cat_%s%s_to_%s%s' % (sub.monomer.name, ''.join(filter(lambda a: a != None, sub.site_conditions.values())),
                                     prod.monomer.name, ''.join(filter(lambda a: a != None, prod.site_conditions.values())))
    r_name_rev = '%s%s_to_%s%s' % (prod.monomer.name, ''.join(filter(lambda a: a != None, prod.site_conditions.values())),
                                         sub.monomer.name, ''.join(filter(lambda a: a != None, sub.site_conditions.values())))

    if (not klist):
        # Default parameter values
        kf = Parameter(r_name_fwd + '_rate', DEFAULT_BI_KF)
        kr = Parameter(r_name_rev + '_rate', DEFAULT_KR)
    else: 
        kf, kr = klist

    # create the site conditions for the complex
    #stmpdict = sub.site_conditions.copy()
    #ptmpdict = prod.site_conditions.copy()
 
    #sub_copy = sub.monomer(s1tmpdict)
    #sub_loc2 = sub.monomer(s2tmpdict)

    # specify the localizations for the monomers
    #sub_loc1.site_conditions[locname] = loc1
    #sub_loc2.site_conditions[locname] = loc2

    # now that we have the complex elements formed we can write the first
    # step rule
    Rule(r_name_fwd, sub + enz >> prod + enz, kf)
    Rule(r_name_rev, prod >> sub, kr)

## Synthesis and Degradation
def synthesize_and_degrade():
    pass

def synthesize_and_degrade_table():
    pass

## Pore assembly
def pore_species(subunit, site1, site2, size):
    """
    Generate a single species representing a homomeric pore, composed
    of <size> copies of <Subunit> bound together in a ring, with bonds
    formed between <site1> of one unit and <site2> of the next.
    """

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

def assemble_pore_sequential(subunit, site1, site2, size, klist):
    """
    Generate rules to chain identical MonomerPatterns <Subunit> into
    increasingly larger pores of up to <size> units, using sites <site1>
    and <site2> to bind the units to each other.
    """
    if size != len(klist):
        raise ValueError("size and len(klist) must be equal")

    subunit = subunit()

    subunit_name = monomer_pattern_label(subunit)
    r_name_pattern = 'assemble_pore_sequential_%s_%%d' % (subunit_name)

    klist_clean = []
    params_created = ComponentSet()
    for i, sublist in enumerate(klist):
        r_name = r_name_pattern % (i + 2)
        if all(isinstance(x, Parameter) for x in sublist):
            kf, kr = sublist
        elif all(isinstance(x, numbers.Real) for x in sublist):
            # if sublist is numbers, generate the Parameters
            kf = Parameter(r_name + '_kf', sublist[0])
            kr = Parameter(r_name + '_kr', sublist[1])
            params_created.add(kf)
            params_created.add(kr)
        else:
            raise ValueError("klist must contain Parameters objects or numbers")
        klist_clean.append([kf, kr])

    rules = ComponentSet()
    for i in range(2, size + 1):
        M = pore_species(subunit, site1, site2, 1)
        S1 = pore_species(subunit, site1, site2, i-1)
        S2 = pore_species(subunit, site1, site2, i)
        r = Rule(r_name_pattern % i, M + S1 <> S2, *klist_clean[i-2])
        rules.add(r)

    return rules | params_created

def transport_pore(subunit, sp_site1, sp_site2, sc_site, min_size, max_size,
                   csource, cdest, c_site, klist):
    """
    Generate rules to transport MonomerPattern <csource> to <cdest> (cargo)
    through any of a series of pores of at least <min_size> and at most
    <max_size> subunits binding on <spsite1> and <spsite2>. Subunit and cargo
    bind at sites scsite and csite, respectively.
    """

    # turn any Monomers into MonomerPatterns
    subunit = subunit()
    csource = csource()
    cdest = cdest()

    # verify that sites are valid
    if sc_site not in csource.monomer.sites_dict:
        raise ValueError("sc_site '%s' not present in csource '%s'" %
                         (sc_site, csource.monomer.name))
    if sc_site not in cdest.monomer.sites_dict:
        raise ValueError("sc_site '%s' not present in cdest '%s'" %
                         (sc_site, cdest.monomer.name))

    subunit_name = monomer_pattern_label(subunit)
    csource_name = monomer_pattern_label(csource)
    cdest_name = monomer_pattern_label(cdest)

    rc_name_pattern = 'transport_complex_%s_pore_%%d_%s' % \
                      (subunit_name, csource_name)
    rd_name_pattern = 'transport_dissociate_%s_pore_%%d_from_%s' % \
                      (subunit_name, cdest_name)

    for i in range(min_size, max_size + 1):
        pore = pore_species(subunit, sp_site1, sp_site2, i)
        # require all pore subunit sites to be empty for match
        for mp in pore.monomer_patterns:
            mp.site_conditions[sc_site] = None

        rc_name = rc_name_pattern % i
        rd_name = rd_name_pattern % i

        rule_rates = rates[i-min_size]
        cpore = pore._copy()
        source_bonds = range(i+1, i+1+i)
        for b in range(i):
            cpore.monomer_patterns[b].site_conditions[sc_site] = \
                                                            source_bonds[b]
        sc_complex = cpore % csource({c_site: source_bonds})
        Rule(rc_name, pore + csource({c_site: None}) <> sc_complex,
             *rule_rates[0:2])
        Rule(rd_name, sc_complex >> pore + cdest({c_site: None}), rule_rates[2])
   
