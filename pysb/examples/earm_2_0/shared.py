"""
Overview
========

This module declares a number of functions and variables that are
used by many of the EARM 2 models. The functions can be divided into the
following four categories:

1. Functions that are specific to the models in EARM 2, but are used by all of
   them. The only macro of this type is

   - :py:func:`observables`

2. Aliases to generalized macros in pysb.macros that provide default values
   for site names or other arguments. Macros of this type include:

   - :py:func:`catalyze`
   - :py:func:`bind`
   - :py:func:`bind_table`
   - :py:func:`assemble_pore_sequential`
   - :py:func:`pore_transport`

3. Macros for mechanisms that appear within the models previously published by
   the research group of Pingping Shen (or the model from [Howells2011]_, which
   is derived from one of Shen's models):

   - :py:func:`assemble_pore_spontaneous`
   - :py:func:`displace`
   - :py:func:`displace_reversibly`

4. Macros for mechanisms that appear within the models described in our
   group's earlier work, specifically the models described in [Albeck2008]_:

   - :py:func:`catalyze_convert`
   - :py:func:`one_step_conv`
   - :py:func:`pore_bind`
"""

# Preliminaries
# =============

# We need the main things from pysb.core plus a few extras:

from pysb import *
from pysb import MonomerPattern, ComplexPattern, ComponentSet
import pysb.macros as macros
from pysb.util import alias_model_components
import functools

# Also pull in Avogadro's number from Scipy:

from scipy.constants import N_A

# Global variables
# ================

# Default constants
# -----------------

# **Default cell volume** to be used when converting between stochastic and
# deterministic rates. The value used is
#
# (1/N_A) * 1e12 L = 1.661e-12 L (1.661 picoliters or 1661 um^3)
#
# is at the lower end of the range for whole cell volume,
# (see `BioNumbers ID 103725
# <http://bionumbers.hms.harvard.edu/bionumber.aspx?&id=103725&ver=14&trm=hela%20cell%20volume>`_).
# but it is slightly above the range given for the volume of HeLa cytoplasm
# (see `BioNumbers ID 106664
# <http://bionumbers.hms.harvard.edu/bionumber.aspx?&id=106664&ver=3&trm=hela%20cell%20volume>`_).
#
# One nice thing about using this value is that since
# 1.661e-12 L * N_A = 1e12 L, it leads to the rule of thumb that
# 1e-9 Molar = 1000 molecules:
#
# [1e-9 Moles/L] = [No. of molecules] / (N_A * vol)
#
# [1e-9 Moles/L] = [No. of molecules] / 1e12 L
#
# [1e-9 Moles/L] * [1e12 L/Moles] = [No. of molecules]
#
# [No. of molecules] = 1e3

V = (1/N_A) * 1e12

# **Default forward and reverse rates** for translocation reactions:

transloc_rates = [1e-2, 1e-2]

# **Rate scaling** for reactions occurring on the mitochondrial membrane.
# `mito_fractional_volume` represents the fractional volume of the
# mitochondrial membrane compartment, so the forward rate constants for
# reactions on the membrane is `1/v`. The approach and the value used is
# adopted from [Albeck2008]_.

mito_fractional_volume = 0.07
rate_scaling_factor = 1./mito_fractional_volume

# Aliases
# -------

# Some useful aliases for typical Bax/Bak states:

active_monomer = {'state':'A', 's1': None, 's2': None}
inactive_monomer = {'state':'C', 's1': None, 's2': None}

# Observables declarations
# ========================

def observables():
    """Declare observables commonly used for the TRAIL pathway.

    Declares truncated (and mitochondrial) Bid, cytosolic (i.e., released)
    Smac, and cleaved PARP.
    """

    alias_model_components()
    # Observables
    # ===========
    Observable('mBid',  Bid(state='M'))
    Observable('aSmac', Smac(state='A'))
    Observable('cPARP', PARP(state='C'))

# Aliases to pysb.macros
# ======================

def catalyze(enz, sub, product, klist):
    """Alias for pysb.macros.catalyze with default binding sites."""

    return macros.catalyze(enz, 'bf', sub, 'bf', product, klist)

def bind(a, b, klist):
    """Alias for pysb.macros.bind with default binding sites."""

    return macros.bind(a, 'bf', b, 'bf', klist)

def bind_table(table, **kwargs):
    """Alias for pysb.macros.bind_table with default binding sites."""

    return macros.bind_table(table, 'bf', 'bf', **kwargs)

def assemble_pore_sequential(subunit, size, klist):
    """Alias for pysb.macros.assemble_pore_sequential with default sites.

    Uses default pore site names as the sites for subunit-subunit binding in
    the pore.
    """

    return macros.assemble_pore_sequential(subunit, 's1', 's2', size, klist)

def pore_transport(subunit, size, csource, cdest, ktable):
    """Alias for pysb.macros.pore_transport with default arguments.

    - Uses the default binding site names for the binding site on the pore
      and on the cargo
    - Uses the default pore site names for subunit-subunit binding
    - Uses only a single size (not a min and max size) for the size of
      transport-competent pores
    """

    return macros.pore_transport(subunit, 's1', 's2', 'bf',
                                 size, size, csource, 'bf', cdest, ktable)

def pore_bind(subunit, size, cargo, klist):
    """Alias for pysb.macros.pore_bind with default arguments.

    - Uses the default binding site names for the binding site on the pore
      and on the cargo
    - Uses the default pore site names for subunit-subunit binding
    """

    return macros.pore_bind(subunit, 's1', 's2', 'bf',
                                 size, cargo, 'bf', klist)


# Macros used by the Shen models
# ==============================

def assemble_pore_spontaneous(subunit, klist):
    """Generate the order-4 assembly reaction 4*Subunit <> Pore."""

    # This is a function that is passed to macros._macro_rule to generate
    # the name for the pore assembly rule. It follows the pattern of,
    # e.g., "BaxA_to_BaxA4" for a Bax pore of size 4.
    def pore_rule_name(rule_expression):
        react_p = rule_expression.reactant_pattern
        mp = react_p.complex_patterns[0].monomer_patterns[0]
        subunit_name = macros._monomer_pattern_label(mp)
        pore_name = mp.monomer.name
        return '%s_to_%s%d' % (subunit_name, mp.monomer.name, 4)

    # Alias for a subunit that is capable of forming a pore
    free_subunit = subunit(s1=None, s2=None)

    # Create the pore formation rule
    macros._macro_rule('spontaneous_pore',
        free_subunit + free_subunit + free_subunit + free_subunit <>
        subunit(s1=1, s2=4) % subunit(s1=2, s2=1) % \
        subunit(s1=3, s2=2) % subunit(s1=4, s2=3),
        klist, ['kf', 'kr'], name_func=pore_rule_name)

def displace(lig1, lig2, target, k):
    """Generate unidirectional displacement reaction L1 + L2:T >> L1:T + L2.

    The signature can be remembered with the following formula:
    "lig1 displaces lig2 from target."
    """

    return macros._macro_rule('displace',
         lig1({'bf':None}) + lig2({'bf':1}) % target({'bf':1}) >>
         lig1({'bf':1}) % target({'bf':1}) + lig2({'bf':None}),
         [k], ['kf'])

def displace_reversibly(lig1, lig2, target, klist):
    """Generate reversible displacement reaction L1 + L2:T <> L1:T + L2.

    The signature can be remembered with the following formula:
    "lig1 displaces lig2 from target." The first rate given in
    in klist specifies the forward rate of this reaction; the second
    specifies the reverse rate.
    """

    return macros._macro_rule('displace',
         lig1({'bf':None}) + lig2({'bf':1}) % target({'bf':1}) <>
         lig1({'bf':1}) % target({'bf':1}) + lig2({'bf':None}),
         klist, ['fwd_kf', 'rev_kf'])

# Macros used by the Albeck models
# ================================

def catalyze_convert(sub1, sub2, product, klist, site='bf'):
    """Automation of the Sub1 + Sub2 <> Sub1:Sub2 >> Prod two-step reaction.

    Because product is created by the function, it must be fully specified.
    """

    # Make sure that the substrates have the site:
    macros._verify_sites(sub1, site)
    macros._verify_sites(sub2, site)

    components = macros._macro_rule('bind',
                             sub1({site: None}) + sub2({site: None}) <>
                             sub1({site: 1}) % sub2({site: 1}),
                             klist[0:2], ['kf', 'kr'])
    components |= macros._macro_rule('convert',
                              sub1({site: 1}) % sub2({site: 1}) >> product,
                              [klist[2]], ['kc'])
    return components

def one_step_conv(sub1, sub2, product, klist, site='bf'):
    """ Bind sub1 and sub2 to form one product: sub1 + sub2 <> product.
    """

    kf, kr = klist

    # Make sure that the substrates have the site:
    macros._verify_sites(sub1, site)
    macros._verify_sites(sub2, site)

    return macros._macro_rule('convert',
                       sub1({site: None}) + sub2({site: None}) <> product,
                       klist, ['kf', 'kr'])

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
        will be created with automatically generated names based on <TODO>
        and these parameters will be included at the end of the returned
        component list.
    """

    macros._verify_sites(subunit, sc_site)
    macros._verify_sites(cargo, c_site)

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
                             macros._monomer_pattern_label(cargo))

    components = ComponentSet()
    # Set up some aliases that are invariant with pore size
    subunit_free = subunit({sc_site: None})
    cargo_free = cargo({c_site: None})

    #for size, klist in zip(range(min_size, max_size + 1), ktable):

    # More aliases which do depend on pore size
    pore_free = macros.pore_species(subunit_free, sp_site1, sp_site2, size)

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
    components |= macros._macro_rule('pore_bind',
                              pore_free + cargo_free <> pc_complex,
                              klist[0:2], ['kf', 'kr'],
                              name_func=name_func)

    return components
