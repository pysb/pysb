"""
Overview
========

Three models of MOMP (:py:func:`direct`, :py:func:`indirect`, and
:py:func:`embedded`), each incorporating a larger repertoire of Bcl-2 family
members than previously published models, including:

* One **activator,** Bid.
* Two **sensitizers,** Bad and Noxa.
* Two **effectors,** Bax and Bak.
* Three **anti-apoptotics**, Bcl-2, Bcl-xL, and Mcl-1.

The Models
----------

Note that in each of the three models, interactions between Bcl-2 proteins only
occur at the mitochondrial membrane. The following are brief descriptions of
each model.

* :py:func:`direct`. In this model, tBid directly activates both Bax and Bak;
  the anti-apoptotics bind tBid and the sensitizers (Bad and Noxa) but not
  Bax and Bak.
* :py:func:`indirect`. Bax and Bak are not explicitly activated by tBid, but
  rather are in an equilibrium between inactive and active states. The
  anti-apoptotics bind tBid, sensitizers, and Bax and Bak.
* :py:func:`embedded`. Combines elements of both direct and indirect: tBid
  activates Bax and Bak; the anti-apoptotics bind tBid, sensitizers and Bax and
  Bak. In addition, Bax and Bak are able to auto-activate.

Organization of models into Motifs
----------------------------------

Because the three models share many aspects, the mechanisms that they share have
been written as small "motifs" implemented as subroutines. These are:

* :py:func:`translocate_tBid_Bax_BclxL`
* :py:func:`tBid_activates_Bax_and_Bak`
* :py:func:`tBid_binds_all_anti_apoptotics`
* :py:func:`sensitizers_bind_anti_apoptotics`
* :py:func:`effectors_bind_anti_apoptotics`
* :py:func:`lopez_pore_formation`

The implementation details of these motifs can be seen by examining the
source code.

Monomer and initial declaration functions
-----------------------------------------

The models share the same set of Monomer and initial condition declarations,
which are contained with the following two functions:

* :py:func:`momp_monomers`
* :py:func:`declare_initial_conditions`
"""

# Preliminaries
# =============
#
# We'll need everything from the pysb core, and some macros:

from pysb import *
from shared import *
from pysb.macros import equilibrate
from pysb.util import alias_model_components

# Globals
# -------

# Default rate constants for catalytic activation (fwd, rev, cat):

activation_rates = [        1e-7, 1e-3, 1] 

# Shared functions
# ================

# Monomer and initial condition declarations
# ------------------------------------------

def momp_monomers():
    """Declare the monomers for the Bcl-2 family proteins, Cyto c, and Smac.

    Annotation() declarations embedded in this function associate UniProt
    identifiers with  each protein.

    'bf' is the site to be used for all binding reactions (with the
    exception of Bax and Bak, which have additional sites used for
    oligomerization).

    The 'state' site denotes various localization and/or activity states of a
    Monomer, with 'C' denoting cytoplasmic localization and 'M' mitochondrial
    localization. Most Bcl-2 proteins have the potential for both cytoplasmic
    and mitochondrial localization, with the exceptions of Bak and Bcl-2,
    which are apparently constitutively mitochondrial.
    """

    # **Activators**.
    # Bid, states: Untruncated, Truncated, truncated and Mitochondrial
    Monomer('Bid', ['bf', 'state'], {'state':['U', 'T', 'M']})

    # **Effectors**
    # Bax, states: Cytoplasmic, Mitochondrial, Active
    # sites 's1' and 's2' are used for pore formation
    Monomer('Bax', ['bf', 's1', 's2', 'state'], {'state':['C', 'M', 'A']})

    # Bak, states: inactive and Mitochondrial, Active (and mitochondrial)
    # sites 's1' and 's2' are used for pore formation
    Monomer('Bak', ['bf', 's1', 's2', 'state'], {'state':['M', 'A']})

    # **Anti-Apoptotics**
    Monomer('Bcl2', ['bf'])
    Monomer('BclxL', ['bf', 'state'], {'state':['C', 'M']})
    Monomer('Mcl1', ['bf', 'state'], {'state':['M', 'C']})

    # **Sensitizers**
    Monomer('Bad', ['bf', 'state'], {'state':['C', 'M']})
    Monomer('Noxa', ['bf', 'state'], {'state': ['C', 'M']})

    # **Cytochrome C and Smac**
    Monomer('CytoC', ['bf', 'state'], {'state':['M', 'C', 'A']})
    Monomer('Smac', ['bf', 'state'], {'state':['M', 'C', 'A']})

    alias_model_components()

    # Annotations
    Annotation(Bid, 'http://identifiers.org/uniprot/P55957')
    Annotation(Bax, 'http://identifiers.org/uniprot/Q07812')
    Annotation(Bak, 'http://identifiers.org/uniprot/Q16611')
    Annotation(Bcl2, 'http://identifiers.org/uniprot/P10415')
    Annotation(BclxL, 'http://identifiers.org/uniprot/Q07817')
    Annotation(Mcl1, 'http://identifiers.org/uniprot/Q07820')
    Annotation(Bad, 'http://identifiers.org/uniprot/Q92934')
    Annotation(Noxa, 'http://identifiers.org/uniprot/Q13794')
    Annotation(CytoC, 'http://identifiers.org/uniprot/P99999')
    Annotation(Smac, 'http://identifiers.org/uniprot/Q9NR28')

def declare_initial_conditions():
    """Declare initial conditions for Bcl-2 family proteins, Cyto c, and Smac.
    """
    Parameter('Bid_0'   , 4.0e4) # Bid
    Parameter('BclxL_0' , 2.0e4) # cytosolic BclxL
    Parameter('Mcl1_0'  , 2.0e4) # Mitochondrial Mcl1
    Parameter('Bcl2_0'  , 2.0e4) # Mitochondrial Bcl2
    Parameter('Bad_0'   , 1.0e3) # Bad
    Parameter('Noxa_0'  , 1.0e3) # Noxa
    Parameter('CytoC_0' , 5.0e5) # cytochrome c
    Parameter('Smac_0'  , 1.0e5) # Smac
    Parameter('Bax_0'   , 0.8e5) # Bax
    Parameter('Bak_0'   , 0.2e5) # Bak

    alias_model_components()

    Initial(Bid(bf=None, state='U'), Bid_0)
    Initial(Bad(bf=None, state='C'), Bad_0)
    Initial(Bax(bf=None, s1=None, s2=None, state='C'), Bax_0)
    Initial(Bak(bf=None, s1=None, s2=None, state='M'), Bak_0)
    Initial(Bcl2(bf=None), Bcl2_0)
    Initial(BclxL (bf=None, state='C'), BclxL_0)
    Initial(Mcl1(bf=None, state='M'), Mcl1_0)
    Initial(Noxa(bf=None, state='C'), Noxa_0)
    Initial(CytoC(bf=None, state='M'), CytoC_0)
    Initial(Smac(bf=None, state='M'), Smac_0)

# Motifs
# ------

def translocate_tBid_Bax_BclxL():
    """tBid, Bax and BclXL translocate to the mitochondrial membrane."""
    equilibrate(Bid(bf=None, state='T'), Bid(bf=None, state='M'), [1e-1, 1e-3])

    free_Bax = Bax(bf=None, s1=None, s2=None) # Alias for readability
    equilibrate(free_Bax(state='C'), free_Bax(state='M'),
                transloc_rates)

    equilibrate(BclxL(bf=None, state='C'), BclxL(bf=None, state='M'),
                transloc_rates)

def tBid_activates_Bax_and_Bak():
    """tBid activates Bax and Bak."""
    catalyze(Bid(state='M'), Bax(state='M'), Bax(state='A'), activation_rates)
    catalyze(Bid(state='M'), Bak(state='M'), Bak(state='A'), activation_rates)

def tBid_binds_all_anti_apoptotics():
    """tBid binds and inhibits Bcl2, Mcl1, and Bcl-XL.

    The entries given in the `bind_table` are dissociation constants taken
    from Certo et al. (see ref). Dissociation constants in Certo et al.
    were published as nanomolar binding affinities; here they are converted
    into units of numbers of molecules by multiplying by `N_A` (Avogadro's
    number) and `V` (a default cell volume, specified in :doc:`shared`.

    The default forward rate represents diffusion limited association
    (1e6 Molar^-1 s^-1) and is converted into units of molec^-1 s^-1 by dividing
    by `N_A*V`.

    Certo, M., Del Gaizo Moore, V., Nishino, M., Wei, G., Korsmeyer, S.,
    Armstrong, S. A., & Letai, A. (2006). Mitochondria primed by death signals
    determine cellular addiction to antiapoptotic BCL-2 family members. Cancer
    Cell, 9(5), 351-365. `doi:10.1016/j.ccr.2006.03.027`
    """
    # Doug Green's "MODE 1" inhibition
    bind_table([[                        Bcl2,  BclxL(state='M'),  Mcl1(state='M')],
                [Bid(state='M'),  66e-9*N_A*V,       12e-9*N_A*V,      10e-9*N_A*V]],
               kf=1e6/(N_A*V))

def sensitizers_bind_anti_apoptotics():
    """Binding of Bad and Noxa to Bcl2, Mcl1, and Bcl-XL.

    See comments on units for :py:func:`tBid_binds_all_anti_apoptotics`.
    """

    bind_table([[                        Bcl2,  BclxL(state='M'),  Mcl1(state='M')],
                [Bad(state='M'),  11e-9*N_A*V,       10e-9*N_A*V,             None],
                [Noxa(state='M'),        None,              None,      19e-9*N_A*V]],
               kf=1e6/(N_A*V))

def effectors_bind_anti_apoptotics():
    """Binding of Bax and Bak to Bcl2, BclxL, and Mcl1.

    Affinities of Bak for Bcl-xL and Mcl-1 are taken from Willis et al.

    Preferential affinity of Bax for Bcl-2 and Bcl-xL were taken from Zhai et
    al.  Bax:Bcl2 and Bax:Bcl-xL affinities were given order of magnitude
    estimates of 10nM.

    See comments on units for :py:func:`tBid_binds_all_anti_apoptotics`.

    Willis, S. N., Chen, L., Dewson, G., Wei, A., Naik, E., Fletcher, J. I.,
    Adams, J. M., et al. (2005). Proapoptotic Bak is sequestered by Mcl-1 and
    Bcl-xL, but not Bcl-2, until displaced by BH3-only proteins. Genes &
    Development, 19(11), 1294-1305. `doi:10.1101/gad.1304105`

    Zhai, D., Jin, C., Huang, Z., Satterthwait, A. C., & Reed, J. C. (2008).
    Differential regulation of Bax and Bak by anti-apoptotic Bcl-2 family
    proteins Bcl-B and Mcl-1. The Journal of biological chemistry, 283(15),
    9580-9586.  `doi:10.1074/jbc.M708426200`
    """

    bind_table([[                            Bcl2,  BclxL(state='M'),         Mcl1],
                [Bax(active_monomer), 10e-9*N_A*V,       10e-9*N_A*V,         None],
                [Bak(active_monomer),        None,       50e-9*N_A*V,  10e-9*N_A*V]],
               kf=1e6/(N_A*V))

def lopez_pore_formation(do_pore_transport=True):
    """ Pore formation and transport process used by all modules.
    """
    alias_model_components()

    # Rates
    pore_max_size = 4
    pore_rates = [[2.040816e-04,  # 1.0e-6/v**2
                   1e-3]] * (pore_max_size - 1)
    pore_transport_rates = [[2.857143e-5, 1e-3, 10]] # 2e-6 / v?

    # Pore formation by effectors
    assemble_pore_sequential(Bax(bf=None, state='A'), pore_max_size, pore_rates)
    assemble_pore_sequential(Bak(bf=None, state='A'), pore_max_size, pore_rates)

    # CytoC, Smac release
    if do_pore_transport:
        pore_transport(Bax(bf=None, state='A'), 4, CytoC(state='M'),
                       CytoC(state='C'), pore_transport_rates)
        pore_transport(Bax(bf=None, state='A'), 4, Smac(state='M'),
                       Smac(state='C'), pore_transport_rates)
        pore_transport(Bak(bf=None, state='A'), 4, CytoC(state='M'),
                       CytoC(state='C'), pore_transport_rates)
        pore_transport(Bak(bf=None, state='A'), 4, Smac(state='M'),
                       Smac(state='C'), pore_transport_rates)

# MOMP model implementations
# ==========================

def embedded(do_pore_transport=True):
    """ Direct and indirect modes of action, occurring at the membrane.
    """
    alias_model_components()

    declare_initial_conditions()

    translocate_tBid_Bax_BclxL()

    tBid_activates_Bax_and_Bak()

    # Autoactivation: Bax and Bak activate their own kind, but only when
    # free (i.e. not part of a pore complex)
    catalyze(Bax(active_monomer), Bax(state='M'), Bax(state='A'),
             activation_rates)
    catalyze(Bak(active_monomer), Bak(state='M'), Bak(state='A'),
             activation_rates)

    # Anti-apoptotics bind activator tBid
    # Doug Green's "MODE 1" inhibition
    tBid_binds_all_anti_apoptotics()

    # Anti-apoptotics bind activated effectors
    # Doug Green's "MODE 2" inhibition
    effectors_bind_anti_apoptotics()

    sensitizers_bind_anti_apoptotics()

    # Bax and Bak form pores by sequential addition and transport CytoC/Smac
    lopez_pore_formation(do_pore_transport=do_pore_transport)

def indirect(do_pore_transport=True):
    """Bax and Bak spontaneously form pores without activation.
       The "activator" tBid binds all of the anti-apoptotics.
    """
    alias_model_components()

    declare_initial_conditions()

    translocate_tBid_Bax_BclxL()

    # Bax and Bak spontaneously become activated
    free_Bax = Bax(bf=None, s1=None, s2=None) # Alias
    free_Bak = Bak(bf=None, s1=None, s2=None) # Alias
    equilibrate(free_Bax(state='M'), free_Bax(state='A'), transloc_rates)
    equilibrate(free_Bak(state='M'), free_Bak(state='A'), transloc_rates)

    # Anti-apoptotics bind activator tBid
    # Doug Green's "MODE 1" inhibition
    tBid_binds_all_anti_apoptotics()

    # Anti-apoptotics bind activated effectors
    # Doug Green's "MODE 2" inhibition
    effectors_bind_anti_apoptotics()

    sensitizers_bind_anti_apoptotics()

    # Bax and Bak form pores by sequential addition
    lopez_pore_formation(do_pore_transport=do_pore_transport)

def direct(do_pore_transport=True):
    """Anti-apoptotics prevent BH3-onlies from activating Bax and Bak.

    Bax and Bak require activation to be able to form pores.
    The anti-apoptotics don't inhibit activated Bax and Bak; their only role
    is to bind BH3-onlies.
    """

    alias_model_components()
    declare_initial_conditions()

    translocate_tBid_Bax_BclxL()


    tBid_activates_Bax_and_Bak()

    # Anti-apoptotics bind activator tBid
    # Doug Green's "MODE 1" inhibition
    tBid_binds_all_anti_apoptotics()

    sensitizers_bind_anti_apoptotics()

    # Bax and Bak form pores by sequential addition
    lopez_pore_formation(do_pore_transport=do_pore_transport)
