from pysb.pattern import SpeciesPatternMatcher, match_complex_pattern
from pysb.examples import robertson, bax_pore, bax_pore_sequential, \
    earm_1_3, kinase_cascade, bngwiki_egfr_simple, move_connected
from pysb.bng import generate_equations
from nose.tools import assert_raises
from pysb import as_complex_pattern, as_reaction_pattern, ANY, WILD, \
    Monomer, Model, Compartment
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
                  bngwiki_egfr_simple, move_connected]:
        yield (check_all_species_generated, model.model)


def test_compartment():
    model = Model()
    Compartment('ec', parent=None, dimension=3)
    Compartment('pm', parent=ec, dimension=2)
    Compartment('cp', parent=pm, dimension=3)
    Compartment('nm', parent=cp, dimension=2)
    Monomer('A', ['a'])
    # `as_complex_pattern(A()) ** pm` would match
    #  `A(a=1) ** ec % B(a=1) ** pm`, but
    # `as_complex_pattern(A() ** pm)` would not match
    assert not match_complex_pattern(
        as_complex_pattern(A() ** pm),
        as_complex_pattern(A()) ** pm
    )
    assert match_complex_pattern(
        as_complex_pattern(A(a=None) ** pm),
        as_complex_pattern(A(a=None)) ** pm
    )
    assert match_complex_pattern(
        as_complex_pattern(A()) ** pm,
        as_complex_pattern(A() ** pm)
    )
    # if the species compartment is not a surface
    # compartment then all monomers have to be in that
    # compartment
    assert match_complex_pattern(
        as_complex_pattern(A() ** ec),
        as_complex_pattern(A()) ** ec
    )
    # `as_complex_pattern(A() ** ec)` would match
    # `A(a=1) ** pm % A(a=1) ** ec`, but
    # `as_complex_pattern(A()) ** ec` would not match
    assert not match_complex_pattern(
        as_complex_pattern(A()) ** ec,
        as_complex_pattern(A() ** ec)
    )
    assert match_complex_pattern(
        (A(a=1) % A(a=1)) ** pm,
        (A(a=1) ** pm) % (A(a=1) ** pm)
    )
    # `(A(a=1) % A(a=1)) ** pm` would match
    # `A(a=1) ** pm % A(a=1) ** ec`, but
    # `(A(a=1) ** pm) % (A(a=1) ** pm)` would not match
    assert not match_complex_pattern(
        (A(a=1) ** pm) % (A(a=1) ** pm),
        (A(a=1) % A(a=1)) ** pm
    )
    assert not match_complex_pattern(
        (A(a=1) ** ec) % (A(a=1) ** pm),
        (A(a=1) % A(a=1)) ** pm
    )
    assert not match_complex_pattern(
        (A(a=1) ** ec) % (A(a=1) ** pm),
        (A(a=1) % A(a=1)) ** ec
    )
    assert not match_complex_pattern(
        (A(a=1) % A(a=1)) ** ec,
        (A(a=1) ** ec) % (A(a=1) ** pm)
    )
    assert match_complex_pattern(
        (A(a=1) % A(a=1)) ** pm,
        (A(a=1) ** ec) % (A(a=1) ** pm)
    )







