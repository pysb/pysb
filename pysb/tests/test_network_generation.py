import numpy as np

from pysb.network_generation import ReactionGenerator
from pysb.bng import generate_equations
from pysb.pattern import SpeciesPatternMatcher, match_complex_pattern
from collections import Counter

from .test_importers import _bngl_location, model_from_bngl


def test_reaction_generation():
    for filename in (#'CaOscillate_Func',
                     #'continue',
                     #'deleteMolecules',
                     #'egfr_net',
                     #'empty_compartments_block',
                     #'gene_expr',
                     #'gene_expr_func',
                     #'gene_expr_simple',
                     #'isomerization',
                     #'localfunc',
                     #'michment',
                     'Motivating_example_cBNGL',
                     #'motor',
                     #'simple_system',
                     #'test_compartment_XML',
                     #'test_setconc',
                     #'test_synthesis_cBNGL_simple',
                     #'test_synthesis_complex',
                     #'test_synthesis_complex_0_cBNGL',
                     #'test_synthesis_complex_source_cBNGL',
                     #'test_synthesis_simple',
                     'toy-jim',
                     #'univ_synth',
                     #'visualize',
                     #'Repressilator',
                     'fceri_ji',
                     #'test_paramname',
                     #'tlmr'
                     ):
        full_filename = _bngl_location(filename)
        yield (compare_native_reactions_to_bng_reactions, full_filename)


def compare_native_reactions_to_bng_reactions(bng_file):
    m = model_from_bngl(bng_file)
    generate_equations(m)
    spm = SpeciesPatternMatcher(m)
    for rule in m.rules:
        reactions = [r for r in m.reactions if rule.name in r['rule']]
        # reactions with the same reactants can have multiple different
        # products depending on the matching between reactant_pattern and
        # the reactants, so we aggregate unique sets of reactants and
        # validate the respective reactions together
        for reactants in set(r['reactants'] for r in reactions):
            validate_reaction([r for r in reactions
                               if r['reactants'] == reactants], m, spm)


def get_matching_patterns(reactant_pattern, species):
    matches = [
            np.where([
                match_complex_pattern(cp, s)
                if s is not None and cp is not None
                else False
                for s in species
            ])
            for cp in reactant_pattern.complex_patterns
    ]
    return [
            np.where([
                match_complex_pattern(cp, s)
                if s is not None and cp is not None
                else False
                for s in species
            ])[0][0]
            for cp in reactant_pattern.complex_patterns
    ]


def validate_reaction(reactions, m, spm):
    if len(reactions[0]['rule']) > 1:
        return

    rg = ReactionGenerator(m.rules[reactions[0]['rule'][0]],
                           reactions[0]['reverse'][0])

    # order reactants such that they match
    cp_order = get_matching_patterns(
        rg.reactant_pattern, [m.species[ix]
                              for ix in reactions[0]['reactants']]
    )

    r = rg.generate_reaction([m.species[reactions[0]['reactants'][cp_idx]]
                              for cp_idx in cp_order])

    products = [spm.match(product, index=True, exact=True)[0]
                for product in r['product_patterns']]
    assert Counter(reactions[0]['products']) == Counter(products)
