from pysb.network_generation import ReactionGenerator
from pysb.bng import generate_equations
from pysb.pattern import SpeciesPatternMatcher
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
        yield (compare_pysb_reactions_to_bng_reactions, full_filename)


def compare_pysb_reactions_to_bng_reactions(bng_file):
    m = model_from_bngl(bng_file)
    generate_equations(m)
    spm = SpeciesPatternMatcher(m)
    for rule in m.rules:
        reactions = [r for r in m.reactions if rule.name in r['rule']]

        rg_forward = ReactionGenerator(rule, False, spm)
        requires_reverse = any(True in r['reverse'] for r in reactions)
        if requires_reverse:
            rg_reverse = ReactionGenerator(rule, True, spm)

        # reactions with the same reactants can have multiple different
        # products depending on the matching between reactant_pattern and
        # the reactants, so we aggregate unique sets of reactants and
        # validate the respective reactions together
        for reactants in set(r['reactants'] for r in reactions):
            reactant_reactions = [r for r in reactions
                                  if r['reactants'] == reactants
                                  and len(r['rule']) == 1]
            reactions_forward = [r for r in reactant_reactions
                                 if not r['reverse'][0]]
            validate_reaction(rg_forward, reactions_forward, m)
            if requires_reverse:
                reactions_reverse = [r for r in reactant_reactions
                                     if r['reverse'][0]]
                validate_reaction(rg_reverse, reactions_reverse, m)


def validate_reaction(rg, reactions_bng, m,):
    if not reactions_bng:
        return

    # we can pick index 0 as reactants should all be the same
    reactions_pysb = rg.generate_reactions(reactions_bng[0]['reactants'], m)

    # reactions_pysb and reactions_bng are not equally ordered so we check
    # that the same number of reactions is generated and that we find at
    # least one match for every reaction
    assert len(reactions_bng) == len(reactions_pysb)
    for rxn_pysb in reactions_pysb:
        assert any(
            Counter(rxn_bng['products']) == Counter(rxn_pysb['products'])
            for rxn_bng in reactions_bng
        )
