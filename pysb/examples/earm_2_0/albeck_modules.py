"""
Overview
========

PySB implementations of the extrinsic apoptosis reaction model version 1.0
(EARM 1.0) originally published in [Albeck2008]_.

This file contains functions that implement the extrinsic pathway in three
modules:

- Receptor ligation to Bid cleavage (:py:func:`rec_to_bid`)
- Mitochondrial Outer Membrane Permeabilization (MOMP, see below)
- Pore transport to effector caspase activation and PARP cleavage
  (:py:func:`pore_to_parp`).

For the (MOMP) segment there are five variants, which correspond to the five
models described in Figure 11 of [Albeck2008]_:

- "Minimal Model" (Figure 11b, :py:func:`albeck_11b`)
- "Model B + Bax multimerization" (Figure 11c, :py:func:`albeck_11c`)
- "Model C + mitochondrial transport" (Figure 11d, :py:func:`albeck_11d`)
- "Current model" (Figure 11e, :py:func:`albeck_11e`)
- "Current model + cooperativity" (Figure 11f, :py:func:`albeck_11f`)
"""

from pysb import *
from pysb.util import alias_model_components
from shared import *
from pysb.macros import equilibrate

# Default forward, reverse, and catalytic rates:

KF = 1e-6
KR = 1e-3
KC = 1

# Monomer declarations
# ====================

def ligand_to_c8_monomers():
    """ Declares ligand, receptor, DISC, Flip, Bar and Caspase 8.

    'bf' is the site to be used for all binding reactions.

    The 'state' site denotes various localization and/or activity states of a
    Monomer, with 'C' denoting cytoplasmic localization and 'M' mitochondrial
    localization.
    """

    Monomer('L', ['bf']) # Ligand
    Monomer('R', ['bf']) # Receptor
    Monomer('DISC', ['bf']) # Death-Inducing Signaling Complex
    Monomer('flip', ['bf'])
    # Caspase 8, states: pro, Active
    Monomer('C8', ['bf', 'state'], {'state':['pro', 'A']})
    Monomer('BAR', ['bf'])

    alias_model_components()

    # == Annotations
    Annotation(L, 'http://identifiers.org/uniprot/P50591')
    Annotation(R, 'http://identifiers.org/uniprot/O14763')
    Annotation(DISC, 'http://identifiers.org/obo.go/GO:0031264')
    Annotation(flip, 'http://identifiers.org/uniprot/O15519')
    Annotation(C8, 'http://identifiers.org/uniprot/Q14790')
    Annotation(BAR, 'http://identifiers.org/uniprot/Q9NZS9')

def momp_monomers():
    """Declare the monomers used in the Albeck MOMP modules."""

    # == Activators
    # Bid, states: Untruncated, Truncated, truncated and Mitochondrial
    Monomer('Bid', ['bf', 'state'], {'state':['U', 'T', 'M']})

    # == Effectors
    # Bax, states: Cytoplasmic, Mitochondrial, Active
    # sites 's1' and 's2' are used for pore formation
    Monomer('Bax', ['bf', 's1', 's2', 'state'], {'state':['C', 'M', 'A']})

    # == Anti-Apoptotics
    Monomer('Bcl2', ['bf'])

    # == Cytochrome C and Smac
    Monomer('CytoC', ['bf', 'state'], {'state':['M', 'C', 'A']})
    Monomer('Smac', ['bf', 'state'], {'state':['M', 'C', 'A']})

    alias_model_components()

    # == Annotations
    Annotation(Bid, 'http://identifiers.org/uniprot/P55957')
    Annotation(Bax, 'http://identifiers.org/uniprot/Q07812')
    Annotation(Bcl2, 'http://identifiers.org/uniprot/P10415')
    Annotation(CytoC, 'http://identifiers.org/uniprot/P99999')
    Annotation(Smac, 'http://identifiers.org/uniprot/Q9NR28')

def apaf1_to_parp_monomers():
    """ Declares CytochromeC, Smac, Apaf-1, the Apoptosome, Caspases 3, 6, 9,
    XIAP and PARP.

    The package variable 'bf' specifies the name of the site to be used
    for all binding reactions.

    The 'state' site denotes various localization and/or activity states of a
    Monomer, with 'C' denoting cytoplasmic localization and 'M' mitochondrial
    localization.
    """

    # Apaf-1 and Apoptosome
    Monomer('Apaf', ['bf', 'state'], {'state':['I', 'A']}) # Apaf-1

    # Apoptosome (activated Apaf-1 + caspase 9)
    Monomer('Apop', ['bf'])

    # Csp 3, states: pro, active, ubiquitinated
    Monomer('C3', ['bf', 'state'], {'state':['pro', 'A', 'ub']})

    # Caspase 6, states: pro-, Active
    Monomer('C6', ['bf', 'state'], {'state':['pro', 'A']})

    # Caspase 9
    Monomer('C9', ['bf'])

    # PARP, states: Uncleaved, Cleaved
    Monomer('PARP', ['bf', 'state'], {'state':['U', 'C']})

    # X-linked Inhibitor of Apoptosis Protein
    Monomer('XIAP', ['bf'])

    alias_model_components()

    # == Annotations
    Annotation(Apaf, 'http://identifiers.org/uniprot/O14727')
    Annotation(Apop, 'http://identifiers.org/obo.go/GO:0043293') 
    Annotation(C3, 'http://identifiers.org/uniprot/P42574')
    Annotation(C6, 'http://identifiers.org/uniprot/P55212')
    Annotation(C9, 'http://identifiers.org/uniprot/P55211')
    Annotation(PARP, 'http://identifiers.org/uniprot/P09874')
    Annotation(XIAP, 'http://identifiers.org/uniprot/P98170')

def all_monomers():
    """Shorthand for calling ligand_to_c8, momp, and apaf1_to_parp macros.

    Internally calls the macros ligand_to_c8_monomers(), momp_monomers(), and
    apaf1_to_parp_monomers() to instantiate the monomers for each portion of the
    pathway.
    """

    ligand_to_c8_monomers()
    momp_monomers()
    apaf1_to_parp_monomers()

# Extrinsic apoptosis module implementations
# ==========================================
#
# These functions implement the upstream (:py:func:`rec_to_bid`) and downstream
# (:py:func:`pore_to_parp`) elements of the extrinsic apoptosis pathway.

def rec_to_bid():
    """Defines the interactions from ligand (e.g. TRAIL) binding to Bid
    activation as per EARM 1.0.

    Uses L, R, DISC, flip, C8, BAR, and Bid monomers and their
    associated parameters to generate the rules that describe Ligand/Receptor
    binding, DISC formation, Caspase-8 activation and
    inhibition by flip and BAR as originally specified in EARM 1.0.

    Declares initial conditions for ligand, receptor, Flip, C8, and Bar.
    """

    # Declare initial conditions for ligand, receptor, Flip, C8, and Bar.
    Parameter('L_0',       3000) # 3000 Ligand corresponds to 50 ng/ml SK-TRAIL
    Parameter('R_0'     ,   200) # 200 TRAIL receptor
    Parameter('flip_0'  , 1.0e2) # Flip
    Parameter('C8_0'    , 2.0e4) # procaspase-8
    Parameter('BAR_0'   , 1.0e3) # Bifunctional apoptosis regulator

    # Needed to recognize the monomer and parameter names in the present scope
    alias_model_components()

    Initial(L(bf=None), L_0)
    Initial(R(bf=None), R_0)
    Initial(flip(bf=None), flip_0)
    Initial(C8(bf=None, state='pro'), C8_0)
    Initial(BAR(bf=None), BAR_0)

    # =====================
    # tBID Activation Rules
    # ---------------------
    #        L + R <--> L:R --> DISC
    #        pC8 + DISC <--> DISC:pC8 --> C8 + DISC
    #        Bid + C8 <--> Bid:C8 --> tBid + C8
    # ---------------------
    catalyze_convert(L(), R(), DISC(bf=None ), [4e-7, KR, 1e-5])
    catalyze(DISC(), C8(state='pro'), C8(state='A'), [KF, KR, KC])
    catalyze(C8(state='A'), Bid(state='U'), Bid(state='T'), [KF, KR, KC])
    # ---------------------
    # Inhibition Rules
    # ---------------------
    #        flip + DISC <-->  flip:DISC  
    #        C8 + BAR <--> BAR:C8
    # ---------------------
    bind(DISC(), flip(), [KF, KR])
    bind(BAR(), C8(state='A'), [KF, KR])

def pore_to_parp():
    """Defines what happens after the pore is activated and Cytochrome C and
    Smac are released.

    Uses CytoC, Smac, Apaf, Apop, C3, C6, C8, C9, PARP, XIAP monomers and their
    associated parameters to generate the rules that describe apoptosome
    formation, XIAP inhibition, activation of caspases (including
    caspase-6-mediated feedback), and cleavage of effector caspase substrates
    as specified in EARM 1.0.

    Declares initial conditions for CytoC, Smac, Apaf-1, Apoptosome, caspases
    3, 6, and 9, XIAP, and PARP.
    """

    # Declare initial conditions:

    Parameter('Apaf_0'  , 1.0e5) # Apaf-1
    Parameter('C3_0'    , 1.0e4) # procaspase-3 (pro-C3)
    Parameter('C6_0'    , 1.0e4) # procaspase-6 (pro-C6)
    Parameter('C9_0'    , 1.0e5) # procaspase-9 (pro-C9)
    Parameter('XIAP_0'  , 1.0e5) # X-linked inhibitor of apoptosis protein
    Parameter('PARP_0'  , 1.0e6) # C3* substrate

    alias_model_components()

    Initial(Apaf(bf=None, state='I'), Apaf_0)
    Initial(C3(bf=None, state='pro'), C3_0)
    Initial(C6(bf=None, state='pro'), C6_0)
    Initial(C9(bf=None), C9_0)
    Initial(PARP(bf=None, state='U'), PARP_0)
    Initial(XIAP(bf=None), XIAP_0)

    # CytoC and Smac activation after release
    # --------------------------------------

    equilibrate(Smac(bf=None, state='C'), Smac(bf=None, state='A'),
                          transloc_rates)

    equilibrate(CytoC(bf=None, state='C'), CytoC(bf=None, state='A'),
                          transloc_rates)

    # Apoptosome formation
    # --------------------
    #   Apaf + cCytoC <-->  Apaf:cCytoC --> aApaf + cCytoC
    #   aApaf + pC9 <-->  Apop
    #   Apop + pC3 <-->  Apop:pC3 --> Apop + C3

    catalyze(CytoC(state='A'), Apaf(state='I'), Apaf(state='A'), [5e-7, KR, KC])
    one_step_conv(Apaf(state='A'), C9(), Apop(bf=None), [5e-8, KR])
    catalyze(Apop(), C3(state='pro'), C3(bf=None, state='A'), [5e-9, KR, KC]) 

    # Apoptosome-related inhibitors
    # -----------------------------
    #   Apop + XIAP <-->  Apop:XIAP  
    #   cSmac + XIAP <-->  cSmac:XIAP  

    bind(Apop(), XIAP(), [2e-6, KR]) 
    bind(Smac(state='A'), XIAP(), [7e-6, KR]) 

    # Caspase reactions
    # -----------------
    # Includes effectors, inhibitors, and feedback initiators:
    #
    #   pC3 + C8 <--> pC3:C8 --> C3 + C8 CSPS
    #   pC6 + C3 <--> pC6:C3 --> C6 + C3 CSPS
    #   XIAP + C3 <--> XIAP:C3 --> XIAP + C3_U CSPS
    #   PARP + C3 <--> PARP:C3 --> CPARP + C3 CSPS
    #   pC8 + C6 <--> pC8:C6 --> C8 + C6 CSPS
    catalyze(C8(state='A'), C3(state='pro'), C3(state='A'), [1e-7, KR, KC])
    catalyze(XIAP(), C3(state='A'), C3(state = 'ub'), [2e-6, KR, 1e-1])
    catalyze(C3(state='A'), PARP(state='U'), PARP(state='C'), [KF, 1e-2, KC])
    catalyze(C3(state='A'), C6(state='pro'), C6(state='A'), [KF, KR, KC])
    catalyze(C6(state='A'), C8(state='pro'), C8(state='A'), [3e-8, KR, KC])

# MOMP module implementations
# ===========================

# Motifs
# ------

# Because several of the models in [Albeck2008]_ overlap, some mechanistic
# aspects have been refactored into the following "motifs", implemented as
# functions:

def Bax_tetramerizes(bax_active_state='A', rate_scaling_factor=1):
    """Creates rules for the rxns Bax + Bax <> Bax2, and Bax2 + Bax2 <> Bax4.

    Parameters
    ----------
    bax_active_state : string: 'A' or 'M'
        The state value that should be assigned to the site "state" for
        dimerization and tetramerization to occur.
    rate_scaling_factor : number
        A scaling factor applied to the forward rate constants for dimerization
        and tetramerization. 
    """

    active_unbound = {'state': bax_active_state, 'bf': None}
    active_bax_monomer = Bax(s1=None, s2=None, **active_unbound)
    bax2 =(Bax(s1=1, s2=None, **active_unbound) %
           Bax(s1=None, s2=1, **active_unbound))
    bax4 =(Bax(s1=1, s2=4, **active_unbound) %
           Bax(s1=2, s2=1, **active_unbound) %
           Bax(s1=3, s2=2, **active_unbound) %
           Bax(s1=4, s2=3, **active_unbound))
    Rule('Bax_dimerization', active_bax_monomer + active_bax_monomer <> bax2,
         Parameter('Bax_dimerization_kf', KF*rate_scaling_factor),
         Parameter('Bax_dimerization_kr', KR))
    # Notes on the parameter values used below:
    #  - The factor 2 is applied to the forward tetramerization rate because
    #    BNG (correctly) divides the provided forward rate constant by 1/2 to
    #    account for the fact that Bax2 + Bax2 is a homodimerization reaction,
    #    and hence the effective rate is half that of an analogous
    #    heterodimerization reaction. However, Albeck et al. used the same
    #    default rate constant of 1e-6 for this reaction as well, therefore it
    #    must be multiplied by 2 in order to match the original model
    #  - BNG apparently applies a scaling factor of 2 to the reverse reaction
    #    rate, for reasons we do not entirely understand. The factor of 0.5 is
    #    applied here to make the rate match the original Albeck ODEs.
    Rule('Bax_tetramerization', bax2 + bax2 <> bax4,
         Parameter('Bax_tetramerization_kf', 2*KF*rate_scaling_factor),
         Parameter('Bax_tetramerization_kr', 0.5*KR))

def Bcl2_binds_Bax1_Bax2_and_Bax4(bax_active_state='A', rate_scaling_factor=1):
    """Creates rules for binding of Bcl2 to Bax monomers and oligomers.

    Parameters
    ----------
    bax_active_state : string: 'A' or 'M'
        The state value that should be assigned to the site "state" for
        the Bax subunits in the pore.
    rate_scaling_factor : number
        A scaling factor applied to the forward rate constants for binding
        between Bax (monomers, oligomers) and Bcl2.
    """

    bind(Bax(state=bax_active_state, s1=None, s2=None), Bcl2,
         [KF*rate_scaling_factor, KR])
    pore_bind(Bax(state=bax_active_state), 's1', 's2', 'bf', 2, Bcl2, 'bf',
         [KF*rate_scaling_factor, KR])
    pore_bind(Bax(state=bax_active_state), 's1', 's2', 'bf', 4, Bcl2, 'bf',
         [KF*rate_scaling_factor, KR])


# Modules
# -------

def albeck_11b(do_pore_transport=True):
    """Minimal MOMP model shown in Figure 11b.

    Features:
        - Bid activates Bax
        - Active Bax is inhibited by Bcl2
        - Free active Bax binds to and transports Smac to the cytosol
    """

    alias_model_components()

    # Set initial conditions
    Initial(Bid(state='U', bf=None), Parameter('Bid_0', 1e5))
    Initial(Bax(bf=None, **inactive_monomer), Parameter('Bax_0', 1e5))
    Initial(Bcl2(bf=None), Parameter('Bcl2_0', 2e4))

    # MOMP Mechanism
    catalyze(Bid(state='T'), Bax(inactive_monomer), Bax(active_monomer),
             [1e-7, KR, KC])
    bind(Bax(active_monomer), Bcl2, [KF, KR])

    # Transport of Smac and Cytochrome C
    if do_pore_transport:
        Initial(Smac(state='M', bf=None), Parameter('Smac_0', 1e6))
        Initial(CytoC(state='M', bf=None), Parameter('CytoC_0', 1e6))
        catalyze(Bax(state='A'), Smac(state='M'), Smac(state='C'),
            [KF, KR, 10])
        catalyze(Bax(state='A'), CytoC(state='M'), CytoC(state='C'),
            [KF, KR, 10])

def albeck_11c(do_pore_transport=True):
    """Model incorporating Bax oligomerization.

    Features:
        - Bid activates Bax
        - Active Bax dimerizes; Bax dimers dimerize to form tetramers
        - Bcl2 binds/inhibits Bax monomers, dimers, and tetramers
        - Bax tetramers bind to and transport Smac to the cytosol
    """

    alias_model_components()
    Initial(Bid(state='U', bf=None), Parameter('Bid_0', 4e4))
    Initial(Bax(bf=None, **inactive_monomer), Parameter('Bax_0', 1e5))
    Initial(Bcl2(bf=None), Parameter('Bcl2_0', 2e4))

    # tBid activates Bax
    catalyze(Bid(state='T'), Bax(inactive_monomer), Bax(active_monomer),
             [1e-7, KR, KC])

    # Bax dimerizes/tetramerizes
    Bax_tetramerizes(bax_active_state='A')

    # Bcl2 inhibits Bax, Bax2, and Bax4
    Bcl2_binds_Bax1_Bax2_and_Bax4(bax_active_state='A')

    if do_pore_transport:
        Initial(Smac(state='M', bf=None), Parameter('Smac_0', 1e6))
        Initial(CytoC(state='M', bf=None), Parameter('CytoC_0', 1e6))
        # NOTE change in KF here from previous model!!!!
        pore_transport(Bax(state='A'), 4, Smac(state='M'), Smac(state='C'),
            [[2*KF, KR, 10]])
        pore_transport(Bax(state='A'), 4, CytoC(state='M'), CytoC(state='C'),
            [[KF, KR, 10]])

def albeck_11d(do_pore_transport=True):
    """Model incorporating mitochondrial transport.

    Features:
        - Bid activates Bax
        - Active Bax translocates to the mitochondria
        - All reactions on the mito membrane have increased association rates
        - Mitochondrial Bax dimerizes; Bax dimers dimerize to form tetramers
        - Bcl2 binds/inhibits Bax monomers, dimers, and tetramers
        - Bax tetramers bind to and transport Smac to the cytosol
    """

    alias_model_components()
    Initial(Bid(state='U', bf=None), Parameter('Bid_0', 4e4))
    Initial(Bax(bf=None, **inactive_monomer), Parameter('Bax_0', 1e5))
    Initial(Bcl2(bf=None), Parameter('Bcl2_0', 2e4))

    # Fractional volume of the mitochondrial membrane compartment
    v = 0.07
    rate_scaling_factor = 1./v

    # tBid activates Bax in the cytosol
    catalyze(Bid(state='T'), Bax(inactive_monomer), Bax(active_monomer),
             [1e-7, KR, KC])

    # Active Bax translocates to the mitochondria
    equilibrate(Bax(bf=None, **active_monomer),
                Bax(bf=None, state='M', s1=None, s2=None),
                [1e-2, 1e-2])

    # Bax dimerizes/tetramerizes
    Bax_tetramerizes(bax_active_state='M',
                     rate_scaling_factor=rate_scaling_factor)

    # Bcl2 inhibits Bax, Bax2, and Bax4
    Bcl2_binds_Bax1_Bax2_and_Bax4(bax_active_state='M',
                                  rate_scaling_factor=rate_scaling_factor)

    if do_pore_transport:
        Initial(Smac(state='M', bf=None), Parameter('Smac_0', 1e6))
        Initial(CytoC(state='M', bf=None), Parameter('CytoC_0', 1e6))
        pore_transport(Bax(state='M'), 4, Smac(state='M'), Smac(state='C'),
            [[rate_scaling_factor*2*KF, KR, 10]])
        pore_transport(Bax(state='M'), 4, CytoC(state='M'), CytoC(state='C'),
            [[KF, KR, 10]])

def albeck_11e(do_pore_transport=True):
    """Model incorporating mitochondrial transport and pore "insertion."

    Features:
        - Bid activates Bax
        - Active Bax translocates to the mitochondria
        - All reactions on the mitochondria have increased association rates
        - Mitochondrial Bax dimerizes; Bax dimers dimerize to form tetramers
        - Bcl2 binds/inhibits Bax monomers, dimers, and tetramers
        - Bax tetramers bind to mitochondrial "sites" and become active pores
        - Active pores bind to and transport Smac to the cytosol
    """
    # Build off of the previous model
    albeck_11d(do_pore_transport=False)

    # Add the "Mito" species, with states "Inactive" and "Active".
    Monomer('Mito', ['bf', 'state'], {'state': ['I', 'A']})
    alias_model_components()
    Initial(Mito(state='I', bf=None), Parameter('Mito_0', 5e5))

    v = 0.07
    rate_scaling_factor = 1./v

    # Add activation of mitochondrial pore sites by Bax4
    pore_bind(Bax(state='M'), 's1', 's2', 'bf', 4, Mito(state='I'), 'bf',
         [KF*rate_scaling_factor, KR])
    Rule('Mito_activation',
         MatchOnce(Bax(state='M', bf=5, s1=1, s2=4) %
                   Bax(state='M', bf=None, s1=2, s2=1) %
                   Bax(state='M', bf=None, s1=3, s2=2) %
                   Bax(state='M', bf=None, s1=4, s2=3) %
                   Mito(state='I', bf=5)) >>
                   Mito(state='A', bf=None),
         Parameter('Mito_activation_kc', KC))

    if do_pore_transport:
        Initial(Smac(state='M', bf=None), Parameter('Smac_0', 1e6))
        Initial(CytoC(state='M', bf=None), Parameter('CytoC_0', 1e6))
        catalyze(Mito(state='A'), Smac(state='M'), Smac(state='C'),
            [rate_scaling_factor*2*KF, KR, 10])
        catalyze(Mito(state='A'), CytoC(state='M'), CytoC(state='C'),
            [rate_scaling_factor*2*KF, KR, 10])

def albeck_11f(do_pore_transport=True):
    """Model as in 11e, but with cooperative assembly of Bax pores.

    Association rate constants for Bax dimerization, tetramerization, and
    insertion are set so that they increase at each step (from 1e-8 to 1e-7 and
    then 1e-6), thereby creating cooperative assembly.

    See also the documentation for albeck_11e().
    """

    albeck_11e(do_pore_transport=do_pore_transport)
    alias_model_components()

    # Set parameter values for cooperative pore formation
    equilibrate_BaxA_to_BaxM_kf.value = 1e-4  # was 1e-2 in 11e
    equilibrate_BaxA_to_BaxM_kr.value = 1e-4  # was 1e-2 in 11e
    Bax_dimerization_kf.value /= 100          # was 1e-6 in 11e
    Bax_tetramerization_kf.value /= 10        # was 1e-6 in 11e


