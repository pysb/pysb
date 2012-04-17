from pysb import *

# This model demonstrates some more advanced macros. It implements
# pore assembly via sequential addition of identical subunits, and
# transport of cargo through the pore.


# macro definitions

def pore_species(Subunit, site1, site2, size):
    """
    Generate a single species representing a homomeric pore, composed
    of <size> copies of <Subunit> bound together in a ring, with bonds
    formed between <site1> of one unit and <site2> of the next.
    """

    if size == 0:
        raise ValueError("size must be an integer greater than 0")
    if size == 1:
        Pore = Subunit({site1: None, site2: None})
    elif size == 2:
        Pore = Subunit({site1: 1, site2: None}) % Subunit({site1: None, site2: 1})
    else:
        # build up a ComplexPattern, starting with a single subunit
        Pore = Subunit({site1: 1, site2: 2})
        for i in range(2, size + 1):
            Pore %= Subunit({site1: i, site2: i % size + 1})
        Pore.match_once = True
    return Pore

def pore_assembly(Subunit, site1, site2, size, rates):
    """
    Generate rules to chain identical MonomerPatterns <Subunit> into
    increasingly larger pores of up to <size> units, using sites
    <site1> and <site2> to bind the units to each other.
    """
    rules = []
    for i in range(2, size + 1):
        M = pore_species(Subunit, site1, site2, 1)
        S1 = pore_species(Subunit, site1, site2, i-1)
        S2 = pore_species(Subunit, site1, site2, i)
        rules.append(Rule('%s_pore_assembly_%d' % (Subunit.monomer.name, i),
                          M + S1 <> S2, *rates[i-2]))
    return rules

def pore_transport(Subunit, ssite1, ssite2, min_size, max_size, CargoSource, CargoDest, tsite, rates):
    """
    Generate rules to transport MonomerPattern <CargoSource> to
    <CargoDest> through any of a series of pores of at least
    <min_size> and at most <max_size> subunits, as defined by
    pore_assembly. Uses site <tsite> on both Subunit and CargoSource
    to bind cargo to ONE Subunit during transport. tsite on all other
    Subunits remains empty.
    """
    assert tsite in CargoSource.monomer.sites_dict, \
        "Required site %s not present in %s as required"%(site, CargoSource.monomer.name)
    assert tsite in CargoDest.monomer.sites_dict, \
        "Required site %s not present in %s as required"%(site, CargoDest.monomer.name)

    for i in range(min_size, max_size+1):
        # require all pore subunit <tsite> sites to be empty for Pore match
        Pore = pore_species(Subunit({tsite: None}), ssite1, ssite2, i)

        r1_name = '%s_pore_%d_transport_%s_cplx' % (CargoSource.monomer.name, i, Subunit.monomer.name)
        r2_name = '%s_pore_%d_transport_%s_dssc' % (CargoSource.monomer.name, i, Subunit.monomer.name)

        rule_rates = rates[i-min_size]
        CPore = Pore.copy()
        tbondnum = i + 1
        CPore.monomer_patterns[0].site_conditions[tsite] = tbondnum
        Complex = CPore % CargoSource({tsite: tbondnum})
        Rule(r1_name, Pore + CargoSource({tsite: None}) <> Complex, *rule_rates[0:2])
        Rule(r2_name, Complex >> Pore + CargoDest({tsite: None}), rule_rates[2])


# ========================================

Model()

Monomer('Bax', ['bh3', 'd2', 't'])
Monomer('Smac', ['loc', 't'], {'loc': ['m','c']})

Parameter('Bax_0', 1)
Parameter('Smac_0', 1)

Initial(Bax(bh3=None, d2=None, t=None), Bax_0)
Initial(Smac(loc='m', t=None), Smac_0)

prefix = 'Bax_pore_assembly_'
assembly_rates = [
    [Parameter(prefix + '2_f', 1), Parameter(prefix + '2_r', 1)],
    [Parameter(prefix + '3_f', 1), Parameter(prefix + '3_r', 1)],
    [Parameter(prefix + '4_f', 1), Parameter(prefix + '4_r', 1)]
    ]

prefix = 'Bax_Smac_transport_'
transport_rates = [
    [Parameter(prefix + '4_f', 1), Parameter(prefix + '4_r', 1), Parameter(prefix + '4_c', 1)]
    ]

# specify t=None so the pore can't fall apart while it's bound to cargo
pore_assembly(Bax(t=None), 'bh3', 'd2', 4, assembly_rates)

pore_transport(Bax(), 'bh3', 'd2', 4, 4, Smac(loc='m'), Smac(loc='c'), 't', transport_rates)
