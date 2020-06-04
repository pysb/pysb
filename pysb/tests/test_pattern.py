from pysb.pattern import SpeciesPatternMatcher, match_complex_pattern
from pysb.examples import robertson, bax_pore, bax_pore_sequential, \
    earm_1_3, kinase_cascade, bngwiki_egfr_simple
from pysb.bng import generate_equations
from nose.tools import assert_raises
from pysb import as_complex_pattern, as_reaction_pattern, ANY, WILD, \
    Monomer, Model, Compartment, MultiState
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


def test_patternmatching_compartments():
    Model()
    A = Monomer('A')
    c = Compartment('c')
    d = Compartment('d')
    assert match_complex_pattern(A() % A() ** c, A() ** c % A() ** c)
    assert match_complex_pattern(A() ** c % A() ** c, A() ** c % A() ** c)
    assert match_complex_pattern((A() % A()) ** c, A() ** c % A() ** c)
    assert match_complex_pattern(A() % A() ** c, A() ** d % A() ** c)
    assert not match_complex_pattern(A() % A() ** c, A() ** d % A() ** d)


def test_patternmatching_multibinding():
    Model()
    A = Monomer('A', sites=['a', 'b'], site_states={'b': ['x', 'y']})
    assert match_complex_pattern(
        A(a=ANY, b='x'), A(a=1, b='y') % A(a=[1, 2], b='x') % A(a=2, b='y'),
        count=True
    ) == 2


def test_patternmatching_multistate():
    Model()
    A = Monomer('A', sites=['a', 'a', 'b'], site_states={'b': ['x', 'y']})
    assert match_complex_pattern(
        as_complex_pattern(A(a=ANY, b='x')),
        A(a=MultiState(1, None), b='y') % A(a=MultiState(1, 2), b='x') %
        A(a=MultiState(2, None), b='y'),
        count=True
    ) == 2
    assert match_complex_pattern(
        as_complex_pattern(A(a=MultiState(ANY, None), b='x')),
        A(a=MultiState(1, None), b='y') % A(a=MultiState(1, 2), b='x') %
        A(a=MultiState(2, None), b='y'),
        count=True
    ) == 0
    assert match_complex_pattern(
        as_complex_pattern(A(a=MultiState(ANY, ANY), b='x')),
        A(a=MultiState(1, None), b='y') % A(a=MultiState(1, 2), b='x') %
        A(a=MultiState(2, None), b='y'),
        count=True
    ) == 2

    assert match_complex_pattern(
        as_complex_pattern(A(a=ANY, b='x')),
        A(a=MultiState(1, None), b='y') % A(a=MultiState(1, None), b='x'),
        count=True
    ) == 1
    assert match_complex_pattern(
        as_complex_pattern(A(a=MultiState(ANY, None), b='x')),
        A(a=MultiState(1, None), b='y') % A(a=MultiState(1, None), b='x'),
        count=True
    ) == 1
    assert match_complex_pattern(
        as_complex_pattern(A(a=MultiState(ANY, ANY), b='x')),
        A(a=MultiState(1, None), b='y') % A(a=MultiState(1, None), b='x'),
        count=True
    ) == 0
