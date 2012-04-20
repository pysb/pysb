import inspect
from pysb import *
import pysb.core

DEFAULT_UNI_KF = 1e6 # In units of sec^-1
DEFAULT_BI_KF = 1e-4 # In units of nM^-1 sec^-1
DEFAULT_KR = 0.1     # In units of sec^-1
DEFAULT_KC = 1       # In units of sec^-1

def alias_model_components(model=None):
    """Make all model components visible as symbols in the caller's global namespace"""
    if model is None:
        model = pysb.core.SelfExporter.default_model
    caller_globals = inspect.currentframe().f_back.f_globals
    components = dict((c.name, c) for c in model.all_components())
    caller_globals.update(components)


def two_step_conv(Sub1, Sub2, Prod, klist, site='bf'):
    """Automation of the Sub1 + Sub2 <> Sub1:Sub2 >> Prod two-step reaction (i.e. dimerization).
    This function assumes that there is a site named 'bf' (bind site for fxn)
    which it uses by default. Site 'bf' need not be passed when calling the function."""

    kf, kr, kc = klist
    
    r1_name = 'cplx_%s%s_%s%s' % (Sub2.monomer.name, ''.join(filter(lambda a: a != None, Sub2.site_conditions.values())),
                                     Sub1.monomer.name, ''.join(filter(lambda a: a != None, Sub1.site_conditions.values())))

    #FIXME: this is a bit dirty but it fixes the problem when prod is a pattern
    if isinstance(Prod, pysb.core.MonomerPattern):
        r2_name = 'cplx_%s_via_%s__%s' % (Prod.monomer.name, Sub1.monomer.name, Sub2.monomer.name)
    elif isinstance(Prod, pysb.core.ComplexPattern):
        r2_name = 'cplx_%s_via_%s__%s' % (("_".join([sub.monomer.name for sub in Prod.monomer_patterns])),
                                          Sub1.monomer.name, Sub2.monomer.name)
    
    assert site in Sub1.monomer.sites_dict, \
        "Required site %s not present in %s as required"%(site, Sub1.monomer.name)
    assert site in Sub2.monomer.sites_dict, \
        "Required site %s not present in %s as required"%(site, Sub2.monomer.name)

    # make the intermediate complex components
    s1tmpdict = Sub1.site_conditions.copy()
    s2tmpdict = Sub2.site_conditions.copy()
    
    s1tmpdict[site] = 1
    s2tmpdict[site] = 1

    Sub1Cplx = Sub1.monomer(s1tmpdict)
    Sub2Cplx = Sub2.monomer(s2tmpdict)

    # add the site to the patterns
    Sub1.site_conditions[site] = None
    Sub2.site_conditions[site] = None

    # now that we have the complex elements formed we can write the first step rule
    Rule(r1_name, Sub1 + Sub2 <> Sub1Cplx % Sub2Cplx, kf, kr)
    
    # and finally the rule for the catalytic transformation
    Rule(r2_name, Sub1Cplx % Sub2Cplx >> Prod, kc)

def simple_dim(Sub, Prod, klist, site='bf'):
    """ Convert two Sub species into one Prod species:
    Sub + Sub <> Prod
    """
    kf, kr = klist
    r1_name = 'dimer_%s_to_%s'%(Sub.monomer.name, Prod.monomer.name)
    assert site in Sub.monomer.sites_dict, \
        "Required site %s not present in %s as required"%(site, Sub.monomer.name)

    # create the sites for the monomers
    Sub.site_conditions[site] = None

    # combine the monomers into a product step rule
    Rule(r1_name, Sub + Sub <> Prod, kf, kr)

def pore_species(Subunit, size):
    """
    Generate a single species representing a homomeric pore, composed
    of <size> copies of <Subunit> bound together in a ring, with bonds
    formed between bh3 of one unit and d2 of the next.
    """

    #FIXME: the sites here are hard-coded and named _bh3_ and _d2_
    #not generic and perhaps misleading?
    if size == 0:
        raise ValueError("size must be an integer greater than 0")
    if size == 1:
        Pore = Subunit(bh3=None, d2=None)
    elif size == 2:
        Pore = Subunit(bh3=1, d2=None) % Subunit(d2=1, bh3=None)
    else:
        # build up a ComplexPattern, starting with a single subunit
        Pore = Subunit(bh3=1, d2=2)
        for i in range(2, size + 1):
            Pore %= Subunit(bh3 = i, d2 = i % size + 1)
        Pore.match_once = True
    return Pore

def pore_assembly(Subunit, size, rates):
    """
    Generate rules to chain identical MonomerPatterns <Subunit> into
    increasingly larger pores of up to <size> units, using sites bh3
    and d2 to bind the units to each other.
    """
    rules = []
    for i in range(2, size + 1):
        M = pore_species(Subunit, 1)
        S1 = pore_species(Subunit, i-1)
        S2 = pore_species(Subunit, i)
        rules.append(Rule('%s_pore_assembly_%d' % (Subunit.monomer.name, i),
                          M + S1 <> S2, *rates[i-2]))
    return rules

def pore_transport(Subunit, Source, Dest, min_size, max_size, rates, site='bf'):
    """
    Generate rules to transport MonomerPattern <Source> to <Dest>
    through any of a series of pores of at least <min_size> and at
    most <max_size> subunits, as defined by pore_assembly.  Implicitly
    uses site 'bf' on both Subunit and Source to bind to each other.
    """
    assert site in Source.monomer.sites_dict, \
        "Required site %s not present in %s as required"%(site, Source.monomer.name)
    assert site in Dest.monomer.sites_dict, \
        "Required site %s not present in %s as required"%(site, Dest.monomer.name)

    for i in range(min_size, max_size+1):
        Pore = pore_species(Subunit, i)
        # require all pore subunit bf sites to be empty for Pore match
        for mp in Pore.monomer_patterns:
            mp.site_conditions[site] = None
        SM = Source.monomer
        ssc = Source.site_conditions
        DM = Dest.monomer
        dsc = Dest.site_conditions

        r1_name = '%s_pore_%d_transport_%s_cplx' % (SM.name, i, Subunit.monomer.name)
        r2_name = '%s_pore_%d_transport_%s_dssc' % (SM.name, i, Subunit.monomer.name)

        rule_rates = rates[i-min_size]
        CPore = Pore._copy()
        source_bonds = range(i+1, i+1+i)
        for b in range(i):
            CPore.monomer_patterns[b].site_conditions[site] = source_bonds[b]
        Complex = CPore % SM(ssc, bf=source_bonds)
        Rule(r1_name, Pore + SM(ssc, bf=None) <> Complex, *rule_rates[0:2])
        Rule(r2_name, Complex >> Pore + DM(dsc, bf=None), rule_rates[2])

def one_step_conv(Sub1, Sub2, Prod, klist, site='bf'):
    """ Convert two Sub species into one Prod species:
    Sub + Sub <> Prod
    """
    kf, kr = klist
    r1_name = 'conv_%s_%s_to_%s'%(Sub1.monomer.name, Sub2.monomer.name, Prod.monomer.name)
    assert site in Sub1.monomer.sites_dict, \
        "Required site %s not present in %s as required"%(site, Sub.monomer.name)
    assert site in Sub2.monomer.sites_dict, \
        "Required site %s not present in %s as required"%(site, Sub.monomer.name)
    # create the sites for the monomers

    Sub1.site_conditions[site] = None
    Sub2.site_conditions[site] = None

    # combine the monomers into a product step rule
    Rule(r1_name, Sub1 + Sub2 <> Prod, kf, kr)


#FIXME: pass klist of sorts?
def simple_bind_table(bindtable, parmlist, lmodel, site='bf'):
    """This assumes that the monomers passed are in their desired state without
    the 'bf' site, which will be used for binding.
    bindtable is a list of lists denoting the reactions between two types of reactants
    as follows:

    bindtable[0]: [                     reactypeA0,       reactypeA1...          reactypeAN]
    bindtable[1]: [                           args,             args...               args)]
    bindtable[2]: [reactypeB0, args, 'parmfamA0B0',    'parmfamA1B0'...    'parmfamANB0'   ]
    bindtable[3]: [reactypeB1, args, 'parmfamA0B1',    'parmfamA1B1'...    'parmfamANB1'   ]

    the variable 'lmodel' is the model passed for local lookup of parameter variables
    """

    # parse the list, extract reactants, products and parameter families
    #first line is one set of reactants
    react0 = bindtable[0]
    react0st = bindtable[1]
    react1 = [row[0] for row in bindtable[2:]]
    react1st = [row[1] for row in bindtable[2:]]

    # Notice this makes intrxns of size/index intrxns[react1][react0]
    intrxns = [row[2:] for row in bindtable[2:]]
    
    # Add the bf sites to the reactant states dict
    # NOTE: this will reset the value if it is already set.
    # Build the prod states dict from react dicts, change bf to 1
    prod0st = []
    prod1st = []
    for d in react0st:
        d[site] = None
        prod0st.append(d.copy())
    for d in react1st:
        d[site] = None
        prod1st.append(d.copy())
    for d in prod0st:
        d[site] = 1
    for d in prod1st:
        d[site] = 1
    
    # loop over interactions
    pc = 0 # parameter counter, cheap way of keeping track of which param set in the list to use
    for i in range(0, len(react1)):
        for j in range(0, len(react0)):
            if intrxns[i][j] is True:
                # get the parameters from the parmlist
                kf = parmlist[pc][0]
                kr = parmlist[pc][1]
                # rule name
                rname = 'cplx_%s_%s' % (react1[i].name, react0[j].name)
                # create the rule
                #print "Generating  %s:%s complex"%(react1[i].name, react0[j].name)
                Rule(rname, react1[i](react1st[i]) + react0[j](react0st[j]) <>
                     react1[i](prod1st[i]) % react0[j](prod0st[j]), 
                     kf, kr)
                pc += 1
    if pc != len(parmlist):
        print "WARNING, unassigned parameters from list", parmlist
        print "Assigned",pc,"parameter pairs from a total of", len(parmlist)

#-------------------------------------------------------------------------
# John B Macro Funcs
#-------------------------------------------------------------------------

def two_step_mod_sites(enz, enz_site, sub, sub_site, prod, klist=None):
    """Automation of the Enz + Sub <> Enz:Sub >> Enz + Prod two-step catalytic reaction.
    This function assumes that there is a site named 'bf' (bind site for fxn)
    which it uses by default. This also assumes Enz returns to its original state.
    In an aim for simplicity, site 'bf' need not be passed when calling the function by
    the reactants, but THE FULL STATE OF THE PRODUCT must be specified"""
    
    # FIXME: this will fail if the argument passed is a Complex object. 
    r1_name = 'cplx_%s%s_%s%s' % (sub.monomer.name, ''.join(filter(lambda a: a != None, sub.site_conditions.values())),
                                     enz.monomer.name, ''.join(filter(lambda a: a != None, enz.site_conditions.values())))
    r2_name = 'diss_%s%s_via_%s%s' % (prod.monomer.name, ''.join(filter(lambda a: a != None, prod.site_conditions.values())),
                                         enz.monomer.name, ''.join(filter(lambda a: a != None, enz.site_conditions.values())))

    # Default names for the parameters
    if (not klist):
        kf = Parameter(r1_name + '_kf', DEFAULT_BI_KF)
        kr = Parameter(r1_name + '_kr', DEFAULT_KR)
        kc = Parameter(r2_name + '_kc', DEFAULT_KC)
    else:
        kf, kr, kc = klist #forward, reverse, catalytic
     
    assert enz_site in enz.monomer.sites_dict, \
        "Required site %s not present in %s as required"%(enz_site, Enz.monomer.name)
    assert sub_site in sub.monomer.sites_dict, \
        "Required site %s not present in %s as required"%(sub_site, Sub.monomer.name)
    #assert prod.is_concrete(), \
    #    "The product %s is not a fully specified (concrete) pattern." % (prod) 

    # make the intermediate complex components
    etmpdict = enz.site_conditions.copy()
    stmpdict = sub.site_conditions.copy()
    
    etmpdict[enz_site] = 1
    stmpdict[sub_site] = 1

    enz_cplx = enz.monomer(etmpdict)
    sub_cplx = sub.monomer(stmpdict)

    # add the 'bf' site to the patterns
    enz.site_conditions[enz_site] = None
    sub.site_conditions[sub_site] = None

    # now that we have the complex elements formed we can write the first step rule
    Rule(r1_name, enz + sub <> enz_cplx % sub_cplx, kf, kr)
    
    # and finally the rule for the catalytic transformation
    Rule(r2_name, enz_cplx % sub_cplx >> enz + prod, kc)

def two_step_mod(Enz, Sub, Prod, klist=None, site='bf'):
    two_step_mod_sites(Enz, site, Sub, site, Prod, klist)


def bind(Sub1, site1, Sub2, site2, klist=None):
    """Automation of the Sub1 + Sub2 <> Sub1:Sub2 one-step complex formation, but allows
    the binding sites of both species to be specified.
    Note that it expects Sub1 and Sub2 to be MonomerPatterns (not Monomers), and site1
    and site2 strings indicating the names of the binding sites.
    klist is list of Parameter objects. If klist is not provided as an argument,
    the bind function will generate default forward and reverse parameters.
    """

    # FIXME: this will fail if the argument passed is a complex, or a Monomer object... 
    r_name = 'cplx_%s%s_%s%s' % (Sub1.monomer.name, ''.join(filter(lambda a: a != None, Sub1.site_conditions.values())),
                                     Sub2.monomer.name, ''.join(filter(lambda a: a != None, Sub2.site_conditions.values())))
    
    assert site1 in Sub1.monomer.sites_dict, \
        "Required site %s not present in %s as required"%(site1, Sub1.monomer.name)
    assert site2 in Sub2.monomer.sites_dict, \
        "Required site %s not present in %s as required"%(site2, Sub2.monomer.name)

    if (not klist):
        # Default parameter values
        # Diffusion limited on rate of 1e6 and offrate of 1e-1 implies 100nM binding
        # FIXME get rid of these magic numbers!!!! Put in a global setting of some kind
        kf = Parameter(r_name + '_kf', DEFAULT_BI_KF)
        kr = Parameter(r_name + '_kr', DEFAULT_KR)
    else: 
        kf, kr = klist

    # create the site conditions for the complex
    s1tmpdict = Sub1.site_conditions.copy()
    s2tmpdict = Sub2.site_conditions.copy()
    
    s1tmpdict[site1] = 1
    s2tmpdict[site2] = 1

    Sub1Cplx = Sub1.monomer(s1tmpdict)
    Sub2Cplx = Sub2.monomer(s2tmpdict)

    # create the sites for the monomers
    Sub1.site_conditions[site1] = None
    Sub2.site_conditions[site2] = None
    # now that we have the complex elements formed we can write the first step rule
    Rule(r_name, Sub1 + Sub2 <> Sub1Cplx % Sub2Cplx, kf, kr)
   
 
def simple_bind(Sub1, Sub2, klist, site='bf'):
    """Automation of the Sub1 + Sub2 <> Sub1:Sub2 one-step complex formation. 
    This function assumes that there is a site named 'bf' which, for simplicity
    need not be passed. Invokes the bind function using the same name for
    each site."""
        
    bind(Sub1, site, Sub2, site, klist)

inhibit = simple_bind #alias for simplebind


def multisite_bind_table(bindtable):
    """This assumes that the monomers passed are in their desired state without
    the sites which will be used for binding.
    bindtable is a list of lists denoting the reactions between two types of reactants
    as follows:

    bindtable[0]: [                     reactypeA0,       reactypeA1...          reactypeAN]
    bindtable[1]: [                           args,             args...               args)]
    bindtable[2]: [reactypeB0, args, (Bs, As, fwdrate, revrate)',             ...                    ]
    bindtable[3]: [reactypeB1, args,                 ,              ...                    ]

    To indicate that no interaction occurs, simply enter None in the bind table
    the variable 'lmodel' is the model passed for local lookup of parameter variables
    """

    # parse the list, extract reactants, products and parameter families
    #first line is one set of reactants
    react_rows = [row[0] for row in bindtable[2:]]
    react_row_states = [row[1] for row in bindtable[2:]]
    react_cols = bindtable[0]
    react_col_states = bindtable[1]

    # Notice this makes intrxns of size/index intrxns[react1][react0]
    intrxns = [row[2:] for row in bindtable[2:]]

    # loop over interactions
    pc = 1 # parameter counter
    rc = 1 # rule counter, easy way of making sure names don't clash #FIXME
    for i in range(0, len(react_rows)):
        for j in range(0, len(react_cols)):
            if intrxns[i][j] is not None:

                # Add the bf sites to the reactant states dict
                # NOTE: this will reset the value if it is already set.
                # Build the prod states dict from react dicts, change bf to 1
                (react_row_site, react_col_site, kf_val, kr_val) = intrxns[i][j]

                # get the parameters from the parmlist
                #kf = parmlist[pc][0]
                #kr = parmlist[pc][1]
                react_row_state = react_row_states[i]
                react_col_state = react_col_states[j]
                prod_col_state = []
                prod_row_state = []
                # The binding sites of reactants should be unbound
                react_row_state[react_row_site] = None
                react_col_state[react_col_site] = None
                # The state of the products should be unchanged except at the site
                prod_row_state = react_row_state.copy()
                prod_col_state = react_col_state.copy()
                prod_row_state[react_row_site] = 1
                prod_col_state[react_col_site] = 1

                # Create the parameters
                kf = Parameter('kf' + str(pc), kf_val)
                kr = Parameter('kr' + str(pc), kr_val)

                # Rule name
                rname = 'cplx_%s_%s_%d' % (react_rows[i].name, react_cols[j].name, rc)
                # Create the rule
                #print "Generating  %s:%s complex"%(react1[i].name, react0[j].name)
                Rule(rname, react_rows[i](react_row_state) + react_cols[j](react_col_state) <>
                            react_rows[i](prod_row_state) % react_cols[j](prod_col_state),
                    kf, kr)
                pc += 1
                rc += 1



def two_state_equilibrium(sub, state1, state2, klist=None, sitename='loc'):
    """Create fwd and reverse rules defining a reversible state transition for
    the monomer given by the MonomerPattern sub from one localization state
    to another.

    Creates two rules with names following the pattern: 'monomer_state1_to_state2' and vice versa.

    The function generates a rule with the name following the pattern 
    --  sub is a MonomerPattern specifying the species that translocates. The localization
        state should not be specified here.
    --  state1 and state2 are strings specifying the names of the locations
        (e.g., 'c' for cytoplasmic, 'm' for mitochondrial)
    --  klist is a list of Parameter objects specifying the forward (state1 to state2)
        and reverse (state2 to state1) rates. If not specified, the parameters are generated
        according to the pattern 'sub_state1_to_state2_rate' and 'sub_state2_to_state1_rate'.
    --  sitename is an optional string specifying the name of the site that describes
        the location. Defaults to 'loc'.
    """

    # FIXME: this will fail if the argument passed is a complex, or a Monomer object... 
    r_name_fwd = '%s%s_%s_%s_to_%s' % (sub.monomer.name,
                                       ''.join(filter(lambda a: a != None, sub.site_conditions.values())),
                                       sitename, state1, state2)
    r_name_rev = '%s%s_%s_%s_to_%s' % (sub.monomer.name,
                                       ''.join(filter(lambda a: a != None, sub.site_conditions.values())),
                                       sitename, state2, state1)
    
    # FIXME: ideally, we should also make sure that the localizations themselves
    # have been declared, not just the name of the loc site
    assert sitename in sub.monomer.sites_dict, \
        "Required site %s not present in %s as required"%(sitename, sub.monomer.name)

    if (not klist):
        # Default parameter values
        # Diffusion limited on rate of 1e6 and offrate of 1e-1 implies 100nM binding
        # FIXME get rid of these magic numbers!!!! Put in a global setting of some kind
        kf = Parameter(r_name_fwd + '_rate', DEFAULT_UNI_KF)
        kr = Parameter(r_name_rev + '_rate', DEFAULT_KR)
    else: 
        kf, kr = klist

    # create the site conditions for the complex
    s1tmpdict = sub.site_conditions.copy()
    s2tmpdict = sub.site_conditions.copy()
 
    sub_state1 = sub.monomer(s1tmpdict)
    sub_state2 = sub.monomer(s2tmpdict)

    # specify the localizations for the monomers
    sub_state1.site_conditions[sitename] = state1
    sub_state2.site_conditions[sitename] = state2

    # now that we have the complex elements formed we can write the first step rule
    Rule(r_name_fwd, sub_state1 >> sub_state2, kf)
    Rule(r_name_rev, sub_state2 >> sub_state1, kr)
   
def direct_catalysis_reversible(sub, enz, prod, klist=None):
    """Create fwd and reverse rules for catalysis of the form:
        A + B -> A + C
            C -> A

    Creates two rules with names following the pattern: 'cat_sub_to_prod' and 'prod_to_sub'.

    The function generates a rule with the name following the pattern 
    --  sub is a MonomerPattern specifying the species that is acted upon.
    --  enz is a MonomerPattern specifying the species that determines the rate of the reaction.
        NO BINDING OCCURS BETWEEN THE SPECIES.
    --  prod is a MonomerPattern specifying the state of the sites of sub after catalysis.
        ANY SITES THAT ARE SPECIFIED IN SUB SHOULD BE SPECIFIED FOR PROD AND VICE VERSA.
    --  klist is a list of Parameter objects specifying the forward (sub to prod) and
        reverse (prod to sub) rates. If not specified, the parameters are generated
        according to the pattern 'cat_sub_to_prod_rate' and 'prod_to_sub_rate'.
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

    # now that we have the complex elements formed we can write the first step rule
    Rule(r_name_fwd, sub + enz >> prod + enz, kf)
    Rule(r_name_rev, prod >> sub, kr)
   
 
#-------------------------------------------------------------------------
# Random little helper funcs that make it easier to interact w the model.
#-------------------------------------------------------------------------

def get_param_num(model, name):
    for i in range(len(model.parameters)):
        if model.parameters[i].name == name:
            print i, model.parameters[i]
            break
    return i

def plotoutput(simout, norm=True):
    """ Assume norm is true for now
    """
    pylab.ion()
    pylab.figure()
    nplots = len(simout.shape[0] -1)
    
    
    for i in range(nplots): #assume simout[0] is time
        pass


        
