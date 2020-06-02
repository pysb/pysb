from pysb.pattern import SpeciesPatternMatcher, match_complex_pattern
from pysb.examples import robertson, bax_pore, bax_pore_sequential, \
    earm_1_3, kinase_cascade, bngwiki_egfr_simple
from pysb.bng import generate_equations
from nose.tools import assert_raises
from pysb import as_complex_pattern, as_reaction_pattern, ANY, WILD, \
    Monomer, Model, Compartment, Tag, MultiState
import collections


def test_species_pattern_matcher():
    # See also SpeciesPatternMatcher doctests

    # Check that SpeciesPatternMatcher raises exception if model has no species
    model = robertson.model
    model.reset_equations()
    assert_raises(Exception, SpeciesPatternMatcher, model)

    model = bax_pore.model
    generate_equations(model)
    spm = SpeciesPatternMatcher(model)
    BAX = model.monomers['BAX']
    sp_sets = spm.species_fired_by_reactant_pattern(
        as_reaction_pattern(BAX(t1=None, t2=None))
    )
    assert len(sp_sets) == 1
    assert len(sp_sets[0]) == 2

    sp_sets = spm.species_fired_by_reactant_pattern(
        as_reaction_pattern(BAX(t1=WILD, t2=ANY))
    )
    assert len(sp_sets) == 1
    assert len(sp_sets[0]) == 10


def test_wildcards():
    a_wild = as_complex_pattern(Monomer('A', ['b'], _export=False)(b=WILD))
    b_mon = as_complex_pattern(Monomer('B', _export=None))

    # B() should not match A(b=WILD)
    assert not match_complex_pattern(b_mon, a_wild)


def check_all_species_generated(model):
    """Check the lists of rules and triggering species match BNG"""
    generate_equations(model)
    spm = SpeciesPatternMatcher(model)

    # Get the rules and species firing them using the pattern matcher
    species_produced = dict()
    for rule, rp_species in spm.rule_firing_species().items():
        species_produced[rule.name] = set()
        for sp_list in rp_species:
            for sp in sp_list:
                species_produced[rule.name].add(
                    model.get_species_index(sp))

    # Get the equivalent dictionary of {rule name: [reactant species]} from BNG
    species_bng = collections.defaultdict(set)
    for rxn in model.reactions:
        for rule in rxn['rule']:
            species_bng[rule].update(rxn['reactants'])

    # Our output should match BNG
    assert species_produced == species_bng


def test_all_species_generated():
    for model in [bax_pore, earm_1_3, bax_pore_sequential, kinase_cascade,
                  bngwiki_egfr_simple]:
        yield (check_all_species_generated, model.model)


def test_match_exact_canonicalization():
    Model()
    volume = Compartment('volume', dimension=3)
    surface = Compartment('surface', dimension=2, parent=volume)
    # assignments aren't really necessary here, but prevent this code from
    # looking like a christmas tree in IDEs
    M1 = Monomer('M1', ['a'])
    M2 = Monomer('M2', ['a', 'b'], {'b': ['a', 'b']})
    M3 = Monomer('M3', ['a'])
    M4 = Monomer('M4', ['a', 'a'], {'a': ['a', 'b']})
    t1 = Tag('Tag1')
    t2 = Tag('Tag2')
    # sorting sites
    assert match_complex_pattern(
        as_complex_pattern(M2(a=None, b='a') ** surface),
        as_complex_pattern(M2(b='a', a=None) ** surface),
        exact=True
    )
    # canonicalization cpt + sorting mps + sorting sites
    assert match_complex_pattern(
        (M1(a=1) ** volume % M2(a=1, b='a')) ** surface,
        M2(b='a', a=2) ** surface % M1(a=2) ** volume,
        exact=True
    )
    # canonicalization cpt
    assert match_complex_pattern(
        (M1(a=1) ** volume % M2(a=1, b='a') ** surface) ** surface,
        M1(a=2) ** volume % M2(a=2, b='a') ** surface,
        exact=True
    )
    # canonicalization cpt
    assert not match_complex_pattern(
        (M1(a=1) % M2(a=1, b='a')) ** surface,
        M1(a=2) ** volume % M2(a=2, b='a') ** surface,
        exact=True
    )
    # canonicalization cpt
    assert not match_complex_pattern(
        (M1(a=1) % M2(a=1, b='a')) ** volume,
        M1(a=2) ** volume % M2(a=2, b='a') ** surface,
        exact=True
    )
    # canonicalization cpt
    assert match_complex_pattern(
        (M1(a=1) % M2(a=1, b='a')) ** surface,
        M1(a=2) ** surface % M2(a=2, b='a') ** surface,
        exact=True
    )
    # canonicalization cpt
    assert match_complex_pattern(
        (M1(a=1) % M2(a=1, b='a')) ** volume,
        M1(a=2) ** volume % M2(a=2, b='a') ** volume,
        exact=True
    )

    # canonicalization cpt + sorting mps + bond_and_state
    assert match_complex_pattern(
        ((M2(a=None, b=('b', 3)) % M1(a=1) ** volume) %
         M2(a=1, b=('a', 3))) ** surface,
        M2(b=('a', 5), a=2) ** surface % M2(b=('b', 5), a=None) ** surface %
        M1(a=2) ** volume,
        exact=True
    )
    # sorting mps + bond_and_state
    assert match_complex_pattern(
        (M2(a=1, b=('a', 3)) ** volume % M2(a=1, b=('a', 5)) ** volume %
         M2(a=2, b=('a', 3)) ** surface % M2(a=2, b=('a', 5)) ** surface),
        (M2(a=2, b=('a', 5)) ** volume % M2(a=2, b=('a', 3)) ** volume %
         M2(a=1, b=('a', 5)) ** surface % M2(a=1, b=('a', 3)) ** surface),
        exact=True
    )
    # sorting mps
    assert match_complex_pattern(
        (M1(a=1) % M3(a=1)) ** volume,
        (M3(a=1) % M1(a=1)) ** volume,
        exact=True
    )
    # tag
    assert match_complex_pattern(
        (M1(a=1) % M1(a=1)) ** volume @ t1,
        (M1(a=1) % M1(a=1)) ** volume @ t1,
        exact=True
    )
    # tag
    assert match_complex_pattern(
        (M1(a=1) % M1(a=1)) ** volume @ t1,
        (M1(a=1) % M1(a=1)) ** volume @ t2,
        exact=True
    )
    # count
    assert match_complex_pattern(
        (M1(a=1) % M1(a=1)) ** volume,
        (M1(a=1) % M1(a=1)) ** volume,
        exact=True, count=True
    ) == 1
    # multistate
    assert match_complex_pattern(
        (M4(a=MultiState(('a', 1), 'a'))
         % M4(a=MultiState(('a', 1), 'a'))) ** volume,
        (M4(a=MultiState(('a', 2), 'a'))
         % M4(a=MultiState(('a', 2), 'a'))) ** volume,
    )
    # multistate
    assert match_complex_pattern(
        (M4(a=MultiState(('a', 1), 'a'))
         % M4(a=MultiState('a', ('a', 1)))) ** volume,
        (M4(a=MultiState('a', ('a', 2)))
         % M4(a=MultiState(('a', 2), 'a'))) ** volume,
        exact=True,
    )
    # multistate
    assert match_complex_pattern(
        (M4(a=MultiState(('a', 1), ('a', 2)))
         % M4(a=MultiState(('a', 2), ('a', 1)))) ** volume,
        (M4(a=MultiState(('a', 1), ('a', 2)))
         % M4(a=MultiState(('a', 1), ('a', 2)))) ** volume,
        exact=True,
    )
    # multistate
    assert not match_complex_pattern(
        (M4(a=MultiState(('a', 1), 'b'))
         % M4(a=MultiState(('b', 1), 'a'))) ** volume,
        (M4(a=MultiState(('a', 1), 'b'))
         % M4(a=MultiState(('a', 1), 'b'))) ** volume,
        exact=True,
    )
