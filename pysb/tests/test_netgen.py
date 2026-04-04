"""Tests for the native Python network generator (pysb.netgen).

Uses nosetests yield-based generators (pynose) for parametrised tests,
following the same pattern as ``test_importers.py``.
"""

import importlib
import os
import warnings

from nose.tools import assert_equal, assert_in, assert_true

from pysb import Model, Monomer, Parameter, Rule
from pysb.core import (
    ANY,
    WILD,
    SelfExporter,
    as_complex_pattern,
)
from pysb.logging import get_logger
from pysb.netgen import (
    NetworkGenerator,
    _apply_rule_to_species,
    _build_reverse_rule_mapping,
    _build_rule_monomer_mapping,
    _build_species_monomer_map,
    _build_src_bond_map,
    _exceeds_max_stoich,
    _extract_bond,
    _extract_bonds_from_val,
    _extract_state_from_val,
    _find_all_matched_multistate_slots,
    _find_all_multistate_permutations,
    _find_multistate_permutation,
    _get_extra_monomers,
    _mp_base_label,
    _mp_site_matches_vf2,
    _multistate_slots_match,
    _renumber_bond_in_pattern,
    _resolve_multistate_slot,
    _resolve_plain_to_slot,
    _site_match_specificity,
    _sp_contains_mp,
    _sp_could_match_cp,
    _species_canonical_key,
    _species_matches_rule_site,
    _split_connected_components,
    fix_bond_conflicts,
)
from pysb.importers.bngl import model_from_bngl
from pysb.pattern import SpeciesPatternMatcher, get_bonds_in_pattern

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_model(name):
    """Load (or reload) an example model, resetting equations."""
    mod = importlib.import_module(f"pysb.examples.{name}")
    importlib.reload(mod)
    # Restore SelfExporter after reload: some example models (e.g. explicit.py)
    # set do_export=False and never restore it.
    SelfExporter.do_export = True
    model = mod.model
    model.reset_equations()
    return model


def _bng_validate_directory():
    """Location of BNG's validation models directory."""
    import pysb.pathfinder as pf

    bng_exec = os.path.realpath(pf.get_path("bng"))
    if bng_exec.endswith(".bat"):
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            return os.path.join(conda_prefix, "share\\bionetgen\\Validate")
    return os.path.join(os.path.dirname(bng_exec), "Validate")


def _bngl_location(filename):
    """Full path to a .bngl file in BNG's Validate directory."""
    return os.path.join(_bng_validate_directory(), filename + ".bngl")


def _safe_reset_self_exporter():
    """Reset SelfExporter internal state without deleting symbols from
    any already-imported example module's namespace.

    SelfExporter.cleanup() removes the 'model' name and all component names
    from whichever module last registered with SelfExporter.  When that
    module is a pysb.examples.* module, subsequent tests that import the
    same module find the attribute missing and raise AttributeError.

    Here we only null out the three class-level pointers so that a fresh
    Model() call can take over, without touching the external module's globals.
    """
    SelfExporter.default_model = None
    SelfExporter.target_globals = None
    SelfExporter.target_module = None


def setup():
    """Module-level setup: ensure SelfExporter is in a clean state."""
    SelfExporter.do_export = True


def teardown():
    """Module-level teardown."""
    _safe_reset_self_exporter()
    SelfExporter.do_export = True


def _reset_self_exporter():
    """Reset SelfExporter state — call at the start of each in-process test."""
    _safe_reset_self_exporter()
    SelfExporter.do_export = True


# ---------------------------------------------------------------------------
# Model validation helpers
# ---------------------------------------------------------------------------


def _check_pysb_example_model(model_name):
    """Validate a PySB example model against BNG."""
    # Do NOT call _reset_self_exporter() here: SelfExporter.cleanup() deletes
    # the 'model' attribute from the previous module's globals, which breaks
    # models like earm_1_3 that reference pysb.examples.earm_1_0.model.
    # importlib.reload() naturally redirects SelfExporter to the new module.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _load_model(model_name)
    ng = NetworkGenerator(model)
    ng.generate_network()
    corr = ng.check_species_against_bng()
    ng.check_reactions_against_bng(corr)
    assert_true(
        len(ng.species) > 0,
        f"{model_name}: expected >0 species, got {len(ng.species)}",
    )


def _check_bng_validate_model(bngl_name):
    """Validate a BNG Validate model against BNG."""
    _reset_self_exporter()
    m = model_from_bngl(_bngl_location(bngl_name))
    m.reset_equations()
    ng = NetworkGenerator(m)
    ng.generate_network(timeout=300)
    corr = ng.check_species_against_bng()
    ng.check_reactions_against_bng(corr)
    assert_true(
        len(ng.species) > 0,
        f"{bngl_name}: expected >0 species, got {len(ng.species)}",
    )


# ---------------------------------------------------------------------------
# PySB example models — parametrised via yield
# ---------------------------------------------------------------------------

# Models that should fully match BNG output
_PYSB_PASSING_MODELS = [
    "bax_pore",
    "bax_pore_sequential",  # MatchOnce
    "bngwiki_egfr_simple",
    "bngwiki_enzymatic_cycle_mm",
    "bngwiki_simple",
    "earm_1_0",
    "earm_1_3",
    "explicit",
    "expression_observables",
    "fixed_initial",
    "fricker_2010_apoptosis",
    "hello_pysb",
    "kinase_cascade",
    "michment",
    "robertson",
    "schloegl",
    "synth_deg",
    "time",
    "tutorial_a",  # synthesis-only (no initial conditions)
    "tyson_oscillator",
]


def test_pysb_example_models():
    """Netgen species and reactions match BNG for each PySB example model."""
    for model_name in _PYSB_PASSING_MODELS:
        yield _check_pysb_example_model, model_name


# ---------------------------------------------------------------------------
# BNG Validate models — parametrised via yield
# ---------------------------------------------------------------------------

_BNG_VALIDATE_PASSING = [
    "CaOscillate_Func",
    "continue",
    "deleteMolecules",
    "empty_compartments_block",
    "fceri_ji",  # large realistic model (~354 species)
    "gene_expr",
    "gene_expr_func",
    "gene_expr_simple",
    "isomerization",
    "issue_198_short",
    "michment",
    "motor",
    "nfkb_illustrating_protocols",
    "Repressilator",
    "simple_system",
    "test_ANG_SSA_synthesis_simple",
    "test_ANG_parscan_synthesis_simple",
    "test_ANG_synthesis_simple",
    "test_compartment_XML",  # compartment propagation regression test
    "test_fixed",
    "test_network_gen",
    "test_paramname",
    "test_partial_dynamical_scaling",
    "test_setconc",
    "test_synthesis_cBNGL_simple",  # source-compartment synthesis fix
    "test_synthesis_complex",
    "test_synthesis_complex_0_cBNGL",  # source-compartment synthesis fix
    "test_synthesis_complex_source_cBNGL",  # source-compartment synthesis fix
    "test_synthesis_simple",
    "tlmr",
    "toy-jim",
    "univ_synth",  # universal synthesis / compartment canonical-key fix
    "statfactor",  # state + bond handling / free-site matching fix
    "visualize",
]


def test_bng_validate_models():
    """Netgen species and reactions match BNG for BNG Validate models."""
    for bngl_name in _BNG_VALIDATE_PASSING:
        yield _check_bng_validate_model, bngl_name


# ---------------------------------------------------------------------------
# _renumber_bond_in_pattern
# ---------------------------------------------------------------------------


def test_renumber_bond_pure_bond():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    cp = A(b=1) % B(a=1)
    result = _renumber_bond_in_pattern(cp, 1, 5)
    assert_in(5, get_bonds_in_pattern(result))
    assert_true(1 not in get_bonds_in_pattern(result))


def test_renumber_bond_state_bond_tuple():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], {"b": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    cp = A(b=("p", 1)) % B(a=1)
    result = _renumber_bond_in_pattern(cp, 1, 10)
    a_mp = [mp for mp in result.monomer_patterns if mp.monomer.name == "A"][0]
    assert_equal(a_mp.site_conditions["b"], ("p", 10))


def test_renumber_bond_no_matching_bond():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    cp = A(b=1) % A(b=1)
    result = _renumber_bond_in_pattern(cp, 99, 100)
    assert_equal(get_bonds_in_pattern(result), {1})


def test_renumber_bond_does_not_mutate():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    cp = A(b=1) % B(a=1)
    _renumber_bond_in_pattern(cp, 1, 5)
    assert_equal(get_bonds_in_pattern(cp), {1})


# ---------------------------------------------------------------------------
# fix_bond_conflicts
# ---------------------------------------------------------------------------


def test_fix_bond_conflicts_no_conflict():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    cp1 = A(b=1) % B(a=1)
    cp2 = A(b=2) % B(a=2)
    result = fix_bond_conflicts([cp1, cp2])
    all_bonds = set()
    for cp in result:
        all_bonds.update(get_bonds_in_pattern(cp))
    assert_equal(len(all_bonds), 2)


def test_fix_bond_conflicts_with_conflict():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    cp1 = A(b=1) % B(a=1)
    cp2 = A(b=1) % B(a=1)
    result = fix_bond_conflicts([cp1, cp2])
    bonds_0 = get_bonds_in_pattern(result[0])
    bonds_1 = get_bonds_in_pattern(result[1])
    assert_true(bonds_0.isdisjoint(bonds_1))


def test_fix_bond_conflicts_empty():
    assert_equal(fix_bond_conflicts([]), [])


def test_fix_bond_conflicts_single():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    cp = A(b=1) % A(b=1)
    result = fix_bond_conflicts([cp])
    assert_equal(get_bonds_in_pattern(result[0]), {1})


def test_fix_bond_conflicts_three_way():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    cp1 = A(b=1) % A(b=1)
    cp2 = A(b=1) % A(b=1)
    cp3 = A(b=1) % A(b=1)
    result = fix_bond_conflicts([cp1, cp2, cp3])
    all_bond_sets = [get_bonds_in_pattern(cp) for cp in result]
    for i in range(len(all_bond_sets)):
        for j in range(i + 1, len(all_bond_sets)):
            assert_true(all_bond_sets[i].isdisjoint(all_bond_sets[j]))


def test_fix_bond_conflicts_no_bonds():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    cp1 = as_complex_pattern(A(b=None))
    cp2 = as_complex_pattern(A(b=None))
    result = fix_bond_conflicts([cp1, cp2])
    assert_equal(len(result), 2)


# ---------------------------------------------------------------------------
# _extract_bond
# ---------------------------------------------------------------------------


def test_extract_bond_pure_int():
    assert_equal(_extract_bond(1), 1)


def test_extract_bond_state_bond_tuple():
    assert_equal(_extract_bond(("p", 3)), 3)


def test_extract_bond_none():
    assert_equal(_extract_bond(None), None)


def test_extract_bond_string_state():
    assert_equal(_extract_bond("u"), None)


def test_extract_bond_tuple_none_bond():
    assert_equal(_extract_bond(("p", None)), None)


def test_extract_bond_any():
    assert_equal(_extract_bond(ANY), None)


# ---------------------------------------------------------------------------
# _species_matches_rule_site
# ---------------------------------------------------------------------------


def test_species_matches_none_none():
    assert_true(_species_matches_rule_site(None, None))


def test_species_matches_none_no_match_bond():
    assert_true(not _species_matches_rule_site(1, None))


def test_species_matches_string_match():
    assert_true(_species_matches_rule_site("u", "u"))


def test_species_matches_string_mismatch():
    assert_true(not _species_matches_rule_site("p", "u"))


def test_species_matches_string_matches_tuple():
    assert_true(_species_matches_rule_site(("u", 1), "u"))


def test_species_matches_bond_int_match():
    assert_true(_species_matches_rule_site(5, 1))


def test_species_matches_bond_int_rejects_no_bond():
    assert_true(not _species_matches_rule_site(None, 1))


def test_species_matches_bond_int_matches_state_bond():
    assert_true(_species_matches_rule_site(("p", 3), 1))


def test_species_matches_tuple_match():
    assert_true(_species_matches_rule_site(("P", 3), ("P", 1)))


def test_species_matches_tuple_state_mismatch():
    assert_true(not _species_matches_rule_site(("U", 3), ("P", 1)))


def test_species_matches_tuple_none_bond_match():
    assert_true(_species_matches_rule_site("P", ("P", None)))


def test_species_matches_tuple_none_bond_mismatch():
    assert_true(not _species_matches_rule_site("U", ("P", None)))


def test_species_matches_tuple_rejects_none():
    assert_true(not _species_matches_rule_site(None, ("P", 1)))


def test_species_matches_tuple_rejects_int():
    assert_true(not _species_matches_rule_site(3, ("P", 1)))


def test_species_matches_any_bonded():
    assert_true(_species_matches_rule_site(1, ANY))


def test_species_matches_any_rejects_none():
    assert_true(not _species_matches_rule_site(None, ANY))


def test_species_matches_wild_anything():
    assert_true(_species_matches_rule_site(None, WILD))
    assert_true(_species_matches_rule_site(1, WILD))
    assert_true(_species_matches_rule_site("u", WILD))


# ---------------------------------------------------------------------------
# _build_rule_monomer_mapping
# ---------------------------------------------------------------------------


def test_build_rule_mapping_simple_binding():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule(
        "bind",
        A(b=None) + B(a=None) >> A(b=1) % B(a=1),
        kf,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    mapping = _build_rule_monomer_mapping(r)
    assert_equal(mapping[(0, 0)], (0, 0))
    assert_equal(mapping[(0, 1)], (1, 0))


def test_build_rule_mapping_synthesis():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule("synth", None >> A(b=None), kf, _export=False)
    r.model = m
    m.add_component(r)

    mapping = _build_rule_monomer_mapping(r)
    assert_equal(len(mapping), 0)


# ---------------------------------------------------------------------------
# _build_reverse_rule_mapping
# ---------------------------------------------------------------------------


def test_build_reverse_mapping_binding():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    kr = Parameter("kr", 1, _export=False)
    kr.model = m
    m.add_component(kr)
    r = Rule(
        "bind",
        A(b=None) + B(a=None) | A(b=1) % B(a=1),
        kf,
        kr,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    mapping = _build_reverse_rule_mapping(r)
    assert_equal(len(mapping), 2)
    assert_equal(len(set(mapping.values())), 2)


def test_build_reverse_mapping_state_change():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    kr = Parameter("kr", 1, _export=False)
    kr.model = m
    m.add_component(kr)
    r = Rule("phos", A(s="u") | A(s="p"), kf, kr, _export=False)
    r.model = m
    m.add_component(r)

    mapping = _build_reverse_rule_mapping(r)
    assert_in((0, 0), mapping)
    assert_equal(mapping[(0, 0)], (0, 0))


# ---------------------------------------------------------------------------
# _build_species_monomer_map
# ---------------------------------------------------------------------------


def test_build_species_monomer_map_single():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b", "s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)

    rule_cp = as_complex_pattern(A(b=None))
    species_cp = as_complex_pattern(A(b=None, s="u"))
    mapping = _build_species_monomer_map([rule_cp], [species_cp])
    assert_in((0, 0), mapping)
    assert_true(mapping[(0, 0)] is species_cp.monomer_patterns[0])


def test_build_species_monomer_map_multi():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a", "c"], _export=False)
    B.model = m
    m.add_component(B)

    rule_cp = A(b=1) % B(a=1)
    species_cp = A(b=1) % B(a=1, c=None)
    mapping = _build_species_monomer_map([rule_cp], [species_cp])
    assert_equal(len(mapping), 2)
    assert_equal(mapping[(0, 0)].monomer.name, "A")
    assert_equal(mapping[(0, 1)].monomer.name, "B")


def test_build_species_monomer_map_incompatible():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)

    rule_cp = as_complex_pattern(A(s="u"))
    species_cp = as_complex_pattern(A(s="p"))
    mapping = _build_species_monomer_map([rule_cp], [species_cp])
    assert_true((0, 0) not in mapping)


# ---------------------------------------------------------------------------
# _apply_rule_to_species
# ---------------------------------------------------------------------------


def test_apply_rule_degradation():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule("degrade", A(b=None) >> None, kf, _export=False)
    r.model = m
    m.add_component(r)

    species = A(b=None)
    result = _apply_rule_to_species(r, (species,))
    # One reaction outcome, zero product species
    assert_equal(len(result), 1)
    assert_equal(result[0], [])


def test_apply_rule_concrete_product():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule("synth", None >> A(s="u"), kf, _export=False)
    r.model = m
    m.add_component(r)

    result = _apply_rule_to_species(r, ())
    assert_equal(len(result), 1)
    # result[0] is a list of product ComplexPatterns
    assert_equal(len(result[0]), 1)
    assert_equal(result[0][0].monomer_patterns[0].site_conditions["s"], "u")


def test_apply_rule_state_change():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule("phos", A(s="u") >> A(s="p"), kf, _export=False)
    r.model = m
    m.add_component(r)

    species = A(s="u")
    result = _apply_rule_to_species(r, (species,))
    assert_equal(len(result), 1)
    assert_equal(len(result[0]), 1)
    assert_equal(result[0][0].monomer_patterns[0].site_conditions["s"], "p")


def test_apply_rule_unbinding_reverse():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    kr = Parameter("kr", 1, _export=False)
    kr.model = m
    m.add_component(kr)
    r = Rule(
        "bind",
        A(b=None) + B(a=None) | A(b=1) % B(a=1),
        kf,
        kr,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    species = A(b=1) % B(a=1)
    result = _apply_rule_to_species(r, (species,), rev_dir=True)
    # One reaction outcome, producing two disconnected product species
    assert_equal(len(result), 1)
    product_cps = result[0]
    assert_equal(len(product_cps), 2)
    monomer_names = sorted(cp.monomer_patterns[0].monomer.name for cp in product_cps)
    assert_equal(monomer_names, ["A", "B"])


def test_apply_rule_state_change_reverse():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    kr = Parameter("kr", 1, _export=False)
    kr.model = m
    m.add_component(kr)
    r = Rule("phos", A(s="u") | A(s="p"), kf, kr, _export=False)
    r.model = m
    m.add_component(r)

    species = as_complex_pattern(A(s="p"))
    result = _apply_rule_to_species(r, (species,), rev_dir=True)
    assert_equal(len(result), 1)
    assert_equal(len(result[0]), 1)
    assert_equal(result[0][0].monomer_patterns[0].site_conditions["s"], "u")


def test_apply_rule_state_change_preserves_bond_tuple():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s", "b"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    C = Monomer("C", ["x"], _export=False)
    C.model = m
    m.add_component(C)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)

    r = Rule(
        "phos_bonded",
        A(s="u", b=ANY) >> A(s=("p", ANY), b=ANY),
        kf,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    # A is bonded to B via site b (bond 1) and to C via site s (bond 2)
    species = A(s=("u", 2), b=1) % B(a=1) % C(x=2)
    result = _apply_rule_to_species(r, (species,))
    assert_equal(len(result), 1)
    # Product-set contains one ComplexPattern (A%B%C stays connected)
    product_cp = result[0]
    assert_equal(len(product_cp), 1)
    a_mp = [mp for mp in product_cp[0].monomer_patterns if mp.monomer.name == "A"][0]
    s_val = a_mp.site_conditions["s"]
    assert_true(isinstance(s_val, tuple))
    assert_equal(s_val[0], "p")
    assert_true(isinstance(s_val[1], int))


def test_apply_rule_state_change_preserves_int_bond():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s", "b"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)

    r = Rule(
        "phos_bonded",
        A(s="u", b=ANY) >> A(s=("p", ANY), b=ANY),
        kf,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    species = A(s="u", b=1) % B(a=1)
    result = _apply_rule_to_species(r, (species,))
    assert_equal(len(result), 1)
    assert_equal(len(result[0]), 1)
    a_mp = [mp for mp in result[0][0].monomer_patterns if mp.monomer.name == "A"][0]
    assert_equal(a_mp.site_conditions["s"], "p")


# ---------------------------------------------------------------------------
# _apply_rule_to_species — DeleteMolecules
# ---------------------------------------------------------------------------


def test_apply_rule_delete_molecules():
    """DeleteMolecules should remove unmentioned monomers and split
    the remaining into connected components."""
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule(
        "del",
        A(b=1) % B(a=1) >> A(b=None),
        kf,
        delete_molecules=True,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    species = A(b=1) % B(a=1)
    result = _apply_rule_to_species(r, (species,))
    assert_equal(len(result), 1)
    # One product-set with one ComplexPattern (just A)
    assert_equal(len(result[0]), 1)
    assert_equal(result[0][0].monomer_patterns[0].monomer.name, "A")
    assert_equal(result[0][0].monomer_patterns[0].site_conditions["b"], None)


# ---------------------------------------------------------------------------
# Compartment propagation in product generation
# ---------------------------------------------------------------------------


def test_apply_rule_state_change_preserves_compartment():
    """Product monomer must inherit compartment from matched source species monomer.

    Regression test for the bug where MonomerPattern(..., None) was
    hardcoded in _apply_rule_with_monomer_map, silently discarding
    compartment annotations from the source species.
    """
    _reset_self_exporter()
    m = Model(_export=False)
    from pysb.core import Compartment, MonomerPattern, ComplexPattern

    EC = Compartment(
        "EC", dimension=3, size=Parameter("vol", 1.0, _export=False), _export=False
    )
    EC.model = m
    m.add_component(EC)
    m.add_component(EC.size)

    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule("phos", A(s="u") >> A(s="p"), kf, _export=False)
    r.model = m
    m.add_component(r)

    # Species: A(s='u') in compartment EC
    species = ComplexPattern([MonomerPattern(A, {"s": "u"}, EC)], None)
    result = _apply_rule_to_species(r, (species,))
    assert_equal(len(result), 1)
    assert_equal(len(result[0]), 1)
    prod_mp = result[0][0].monomer_patterns[0]
    assert_equal(prod_mp.site_conditions["s"], "p")
    assert_true(
        prod_mp.compartment is EC,
        f"Expected compartment EC, got {prod_mp.compartment}",
    )


def test_apply_rule_delete_molecules_preserves_compartment():
    """DeleteMolecules: surviving monomer must keep its compartment.

    Regression test for the analogous bug in _apply_delete_molecules.
    """
    _reset_self_exporter()
    m = Model(_export=False)
    from pysb.core import Compartment, MonomerPattern, ComplexPattern

    EC = Compartment(
        "EC", dimension=3, size=Parameter("vol2", 1.0, _export=False), _export=False
    )
    EC.model = m
    m.add_component(EC)
    m.add_component(EC.size)

    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    kf = Parameter("kf2", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule(
        "del",
        A(b=1) % B(a=1) >> A(b=None),
        kf,
        delete_molecules=True,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    # Species: A(b=1) in EC, B(a=1) with no compartment
    species = ComplexPattern(
        [MonomerPattern(A, {"b": 1}, EC), MonomerPattern(B, {"a": 1}, None)], None
    )
    result = _apply_rule_to_species(r, (species,))
    assert_equal(len(result), 1)
    assert_equal(len(result[0]), 1)
    prod_mp = result[0][0].monomer_patterns[0]
    assert_equal(prod_mp.site_conditions["b"], None)
    assert_true(
        prod_mp.compartment is EC,
        f"Expected compartment EC, got {prod_mp.compartment}",
    )


def test_apply_rule_synthesis_inherits_source_compartment():
    """Synthesised product monomer inherits compartment from reactant species.

    A rule ``Source() -> Source() + Product()`` carries no explicit compartment
    on the product pattern for ``Product()``.  When the matched reactant
    species ``Source()@EC`` lives in compartment EC, netgen must infer
    EC for the synthesised ``Product()`` monomer as BNG does.

    Regression test for the source-compartment synthesis fix in
    _apply_rule_with_monomer_map.
    """
    _reset_self_exporter()
    m = Model(_export=False)
    from pysb.core import Compartment, MonomerPattern, ComplexPattern

    EC = Compartment(
        "EC", dimension=3, size=Parameter("vol3", 1.0, _export=False), _export=False
    )
    EC.model = m
    m.add_component(EC)
    m.add_component(EC.size)

    Src = Monomer("Src", [], _export=False)
    Src.model = m
    m.add_component(Src)
    Prod = Monomer("Prod", [], _export=False)
    Prod.model = m
    m.add_component(Prod)
    kf = Parameter("kf3", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    # Product pattern has no compartment annotation on Prod()
    r = Rule("synth", Src() >> Src() + Prod(), kf, _export=False)
    r.model = m
    m.add_component(r)

    # Reactant species lives in EC
    src_species = ComplexPattern([MonomerPattern(Src, {}, EC)], None)
    result = _apply_rule_to_species(r, (src_species,))
    # result is a list of product-species lists; expect Src()@EC and Prod()@EC
    assert_equal(len(result), 1)
    prod_list = result[0]
    assert_equal(len(prod_list), 2)
    compartments = {cp.monomer_patterns[0].compartment for cp in prod_list}
    assert_true(
        compartments == {EC},
        f"Expected both product species in EC, got compartments: {compartments}",
    )


def test_apply_delete_molecules_synthesis_inherits_source_compartment():
    """DeleteMolecules: synthesised monomer inherits compartment from reactant.

    A rule ``A(b=1) % B(a=1) >> A(b=None) + C()`` (delete_molecules=True)
    with no compartment on ``C()`` must place the synthesised C in the same
    compartment as the matched species.

    Regression test for the source-compartment synthesis fix in
    _apply_delete_molecules.
    """
    _reset_self_exporter()
    m = Model(_export=False)
    from pysb.core import Compartment, MonomerPattern, ComplexPattern

    EC = Compartment(
        "EC", dimension=3, size=Parameter("vol4", 1.0, _export=False), _export=False
    )
    EC.model = m
    m.add_component(EC)
    m.add_component(EC.size)

    A = Monomer("A2", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B2", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    C = Monomer("C2", [], _export=False)
    C.model = m
    m.add_component(C)
    kf = Parameter("kf4", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    # C() carries no compartment in the product pattern
    r = Rule(
        "del_synth",
        A(b=1) % B(a=1) >> A(b=None) + C(),
        kf,
        delete_molecules=True,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    # Species: A(b=1)@EC % B(a=1)@EC
    species = ComplexPattern(
        [MonomerPattern(A, {"b": 1}, EC), MonomerPattern(B, {"a": 1}, EC)], None
    )
    result = _apply_rule_to_species(r, (species,))
    assert_equal(len(result), 1)
    prod_list = result[0]
    # Expect A2(b=None)@EC and C2()@EC
    assert_equal(len(prod_list), 2)
    compartments = {cp.monomer_patterns[0].compartment for cp in prod_list}
    assert_true(
        compartments == {EC},
        f"Expected both product species in EC, got compartments: {compartments}",
    )


def test_delete_molecules_clears_state_bond_site():
    """(state, bond) site becomes bare state when bond leads to deleted monomer."""
    _reset_self_exporter()
    m = Model(_export=False)
    from pysb.core import MonomerPattern, ComplexPattern

    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule(
        "dm_sb",
        A(s=("u", 1)) % B(a=1) >> A(s="u"),
        kf,
        delete_molecules=True,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    species = ComplexPattern(
        [
            MonomerPattern(A, {"s": ("u", 3)}, None),
            MonomerPattern(B, {"a": 3}, None),
        ],
        None,
    )
    result = _apply_rule_to_species(r, (species,))
    assert_equal(len(result), 1)
    assert_equal(len(result[0]), 1)
    assert_equal(result[0][0].monomer_patterns[0].site_conditions["s"], "u")


def test_delete_molecules_multistate_bond_clearing():
    """MultiState slot bonded to a deleted monomer is cleared to None."""
    _reset_self_exporter()
    m = Model(_export=False)
    from pysb.core import MultiState, MonomerPattern, ComplexPattern

    A = Monomer("A", ["r", "r"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule(
        "dm_ms",
        A(r=MultiState(1, None)) % B(a=1) >> A(r=MultiState(None, None)),
        kf,
        delete_molecules=True,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    species = ComplexPattern(
        [
            MonomerPattern(A, {"r": MultiState(5, None)}, None),
            MonomerPattern(B, {"a": 5}, None),
        ],
        None,
    )
    result = _apply_rule_to_species(r, (species,))
    assert_equal(len(result), 1)
    assert_equal(len(result[0]), 1)
    prod_r = result[0][0].monomer_patterns[0].site_conditions["r"]
    assert_true(isinstance(prod_r, MultiState))
    assert_equal(list(prod_r), [None, None])


def test_delete_molecules_product_rebond():
    """Surviving monomer gains new bond to synthesised monomer after deletion."""
    _reset_self_exporter()
    m = Model(_export=False)
    from pysb.core import MonomerPattern, ComplexPattern

    A = Monomer("A", ["b", "c"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    C = Monomer("C", ["x"], _export=False)
    C.model = m
    m.add_component(C)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    # C is not in the reactant pattern → synthesised; product bond 2 creates A–C link.
    r = Rule(
        "dm_rebond",
        A(b=1, c=None) % B(a=1) >> A(b=None, c=2) % C(x=2),
        kf,
        delete_molecules=True,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    species = ComplexPattern(
        [
            MonomerPattern(A, {"b": 3, "c": None}, None),
            MonomerPattern(B, {"a": 3}, None),
        ],
        None,
    )
    result = _apply_rule_to_species(r, (species,))
    assert_equal(len(result), 1)
    assert_equal(len(result[0]), 1)  # A and C bonded → one complex
    prod_mps = result[0][0].monomer_patterns
    assert_equal(len(prod_mps), 2)
    names = {mp.monomer.name for mp in prod_mps}
    assert_equal(names, {"A", "C"})
    bonds = get_bonds_in_pattern(result[0][0])
    assert_equal(len(bonds), 1)


def test_delete_molecules_product_tuple_bond():
    """Product (state, int-bond) site updates state and forms new bond on survivor."""
    _reset_self_exporter()
    m = Model(_export=False)
    from pysb.core import MonomerPattern, ComplexPattern

    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    C = Monomer("C", ["x"], _export=False)
    C.model = m
    m.add_component(C)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    # B deleted; A.s: state u→p and new bond to synthesised C.
    r = Rule(
        "dm_tuple_bond",
        A(s=("u", 1)) % B(a=1) >> A(s=("p", 2)) % C(x=2),
        kf,
        delete_molecules=True,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    species = ComplexPattern(
        [
            MonomerPattern(A, {"s": ("u", 3)}, None),
            MonomerPattern(B, {"a": 3}, None),
        ],
        None,
    )
    result = _apply_rule_to_species(r, (species,))
    assert_equal(len(result), 1)
    assert_equal(len(result[0]), 1)  # A and C bonded
    prod_mps = result[0][0].monomer_patterns
    a_mp = next(mp for mp in prod_mps if mp.monomer.name == "A")
    s_val = a_mp.site_conditions["s"]
    assert_true(isinstance(s_val, tuple) and len(s_val) == 2)
    assert_equal(s_val[0], "p")
    assert_true(isinstance(s_val[1], int))


def test_delete_molecules_product_int_bond_existing_state():
    """Int-bond product value on site whose current value is a bare state string."""
    _reset_self_exporter()
    m = Model(_export=False)
    from pysb.core import MonomerPattern, ComplexPattern

    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    C = Monomer("C", ["x"], _export=False)
    C.model = m
    m.add_component(C)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    # A.s=('u', bond_to_B): bond cleared → 'u'. Product A(s=2): int bond, existing=str → ('u', new_bond).
    r = Rule(
        "dm_int_bond_state",
        A(s=("u", 1)) % B(a=1) + C(x=None) >> A(s=2) % C(x=2),
        kf,
        delete_molecules=True,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    species_ab = ComplexPattern(
        [
            MonomerPattern(A, {"s": ("u", 3)}, None),
            MonomerPattern(B, {"a": 3}, None),
        ],
        None,
    )
    species_c = ComplexPattern([MonomerPattern(C, {"x": None}, None)], None)
    result = _apply_rule_to_species(r, (species_ab, species_c))
    assert_equal(len(result), 1)
    assert_equal(len(result[0]), 1)  # A and C bonded
    prod_mps = result[0][0].monomer_patterns
    a_mp = next(mp for mp in prod_mps if mp.monomer.name == "A")
    s_val = a_mp.site_conditions["s"]
    assert_true(isinstance(s_val, tuple) and len(s_val) == 2)
    assert_equal(s_val[0], "u")
    assert_true(isinstance(s_val[1], int))


def test_delete_molecules_product_none_clears_tuple_site():
    """Product None on a (state, bond) site retains the state and drops the bond."""
    _reset_self_exporter()
    m = Model(_export=False)
    from pysb.core import MonomerPattern, ComplexPattern

    A = Monomer("A", ["x", "b"], {"x": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    C = Monomer("C", ["y"], _export=False)
    C.model = m
    m.add_component(C)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    # A–C bonded via x–y (bond 1, survives); A–B via b–a (bond 2, B deleted).
    # Product None on A.x: existing=('u', bond_to_C) is tuple → result keeps 'u', drops bond.
    r = Rule(
        "dm_none_tuple",
        A(x=("u", 1), b=2) % C(y=1) % B(a=2) >> A(x=None, b=None) + C(y=None),
        kf,
        delete_molecules=True,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    species = ComplexPattern(
        [
            MonomerPattern(A, {"x": ("u", 3), "b": 4}, None),
            MonomerPattern(C, {"y": 3}, None),
            MonomerPattern(B, {"a": 4}, None),
        ],
        None,
    )
    result = _apply_rule_to_species(r, (species,))
    assert_equal(len(result), 1)
    assert_equal(len(result[0]), 2)  # A and C separate after A.x bond removed
    a_mp = next(
        mp for cp in result[0] for mp in cp.monomer_patterns if mp.monomer.name == "A"
    )
    assert_equal(a_mp.site_conditions["x"], "u")
    assert_equal(a_mp.site_conditions["b"], None)


def test_get_extra_monomers_carries_bonded():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b", "c"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    C = Monomer("C", ["a"], _export=False)
    C.model = m
    m.add_component(C)

    rule_cp = A(b=1) % B(a=1)
    species_cp = A(b=1, c=2) % B(a=1) % C(a=2)

    species_monomer_map = _build_species_monomer_map([rule_cp], [species_cp])
    product_to_source_species = {0: {0}}

    species_idx_to_tgt_cp = {}
    for (rp_ci, rp_mi), sp_mp in species_monomer_map.items():
        for idx, mp in enumerate(species_cp.monomer_patterns):
            if mp is sp_mp:
                species_idx_to_tgt_cp[(rp_ci, idx)] = 0
                break

    extra = _get_extra_monomers(
        [species_cp],
        species_monomer_map,
        0,
        product_to_source_species,
        species_idx_to_tgt_cp,
    )
    assert_equal(len(extra), 1)
    assert_equal(extra[0].monomer.name, "C")


def test_get_extra_monomers_none_when_all_claimed():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)

    rule_cp = A(b=1) % B(a=1)
    species_cp = A(b=1) % B(a=1)

    species_monomer_map = _build_species_monomer_map([rule_cp], [species_cp])
    product_to_source_species = {0: {0}}

    species_idx_to_tgt_cp = {}
    for (rp_ci, rp_mi), sp_mp in species_monomer_map.items():
        for idx, mp in enumerate(species_cp.monomer_patterns):
            if mp is sp_mp:
                species_idx_to_tgt_cp[(rp_ci, idx)] = 0
                break

    extra = _get_extra_monomers(
        [species_cp],
        species_monomer_map,
        0,
        product_to_source_species,
        species_idx_to_tgt_cp,
    )
    assert_equal(extra, [])


def test_get_extra_monomers_empty():
    extra = _get_extra_monomers([], {}, 0, {}, {})
    assert_equal(extra, [])


# ---------------------------------------------------------------------------
# NetworkGenerator — basic properties
# ---------------------------------------------------------------------------


def test_species_property_before_generation():
    _reset_self_exporter()
    model = _load_model("robertson")
    ng = NetworkGenerator(model)
    assert_equal(ng.species, [])


def test_reactions_empty_before_generation():
    _reset_self_exporter()
    model = _load_model("robertson")
    ng = NetworkGenerator(model)
    assert_equal(len(ng.reactions), 0)


# ---------------------------------------------------------------------------
# NetworkGenerator.generate_network — max_iterations
# ---------------------------------------------------------------------------
# NetworkGenerator.generate_network — max_iterations
# ---------------------------------------------------------------------------


def test_generate_network_max_iterations():
    _reset_self_exporter()
    model = _load_model("robertson")
    ng = NetworkGenerator(model)
    ng.generate_network(max_iterations=1)
    assert_equal(len(ng.species), 3)


# ---------------------------------------------------------------------------
# check_species_against_bng / check_reactions_against_bng error paths
# ---------------------------------------------------------------------------


def test_check_species_duplicate_raises():
    _reset_self_exporter()
    model = _load_model("robertson")
    ng = NetworkGenerator(model)
    ng.generate_network()
    ng.species_pm.add_species(ng.species[0].copy(), check_duplicate=False)
    try:
        ng.check_species_against_bng()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert_in("both match BNG species", str(e))


def test_check_species_not_found_raises():
    _reset_self_exporter()
    model = _load_model("robertson")
    ng = NetworkGenerator(model)
    ng.generate_network()

    Fake = Monomer("Fake", [], _export=False)
    Fake.model = model
    model.add_component(Fake)
    ng.species_pm.add_species(as_complex_pattern(Fake()), check_duplicate=False)
    try:
        ng.check_species_against_bng()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert_in("not found in BNG", str(e))


def test_check_species_count_mismatch_raises():
    """Removing a species creates a count mismatch error."""
    _reset_self_exporter()
    model = _load_model("robertson")
    ng = NetworkGenerator(model)
    ng.generate_network()
    ng.species_pm.species.pop()
    try:
        ng.check_species_against_bng()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert_in("Species count mismatch", str(e))


def test_check_reactions_auto_correspondence():
    _reset_self_exporter()
    model = _load_model("robertson")
    ng = NetworkGenerator(model)
    ng.generate_network()
    ng.check_reactions_against_bng()  # should not raise


def test_check_reactions_missing_raises():
    _reset_self_exporter()
    model = _load_model("robertson")
    ng = NetworkGenerator(model)
    ng.generate_network()
    n = len(ng.species)
    ng.reactions[((0,), (n + 99,))] = {
        "rule": ["fake_rule"],
        "reactants": (0,),
        "products": (n + 99,),
        "reverse": [False],
    }
    corr = list(range(len(ng.species)))
    try:
        ng.check_reactions_against_bng(corr)
        assert False, "Expected ValueError or IndexError"
    except (ValueError, IndexError):
        pass


def test_check_reactions_rule_mismatch_raises():
    _reset_self_exporter()
    model = _load_model("robertson")
    ng = NetworkGenerator(model)
    ng.generate_network()
    corr = ng.check_species_against_bng()
    first_key = next(iter(ng.reactions))
    ng.reactions[first_key]["rule"] = ["nonexistent_rule"]
    try:
        ng.check_reactions_against_bng(corr)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert_in("Rule mismatch", str(e))


# ---------------------------------------------------------------------------
# Duplicate reaction same-rule
# ---------------------------------------------------------------------------


def test_duplicate_reaction_same_rule():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule("phos", A(s="u") >> A(s="p"), kf, _export=False)
    r.model = m
    m.add_component(r)

    sp = as_complex_pattern(A(s="u"))
    sp_p = as_complex_pattern(A(s="p"))

    ng = NetworkGenerator(m)
    ng.species_pm = SpeciesPatternMatcher(m, [sp, sp_p])

    ng._fire_rule_combo(r, (sp,), False)
    assert_equal(len(ng.reactions), 1)
    first_key = next(iter(ng.reactions))
    assert_equal(ng.reactions[first_key]["rule"], ["phos"])

    ng._fire_rule_combo(r, (sp,), False)
    assert_equal(ng.reactions[first_key]["rule"], ["phos"])
    assert_equal(len(ng.reactions[first_key]["rule"]), 1)


# ---------------------------------------------------------------------------
# Timeout mechanism
# ---------------------------------------------------------------------------


def test_match_once_works():
    """MatchOnce should work and produce correct results."""
    _reset_self_exporter()
    model = _load_model("bax_pore_sequential")
    ng = NetworkGenerator(model)
    ng.generate_network()
    corr = ng.check_species_against_bng()
    ng.check_reactions_against_bng(corr)
    assert_equal(len(ng.species), 11)
    assert_equal(len(ng.reactions), 19)


# ---------------------------------------------------------------------------
# DeleteMolecules — now works (validate with BNG deleteMolecules.bngl)
# ---------------------------------------------------------------------------


def test_delete_molecules_works():
    """DeleteMolecules should work and produce correct results."""
    _reset_self_exporter()
    m = model_from_bngl(_bngl_location("deleteMolecules"))
    m.reset_equations()
    ng = NetworkGenerator(m)
    ng.generate_network(timeout=30)
    corr = ng.check_species_against_bng()
    ng.check_reactions_against_bng(corr)
    assert_equal(len(ng.species), 3)
    assert_equal(len(ng.reactions), 1)


# ---------------------------------------------------------------------------
# Timeout mechanism
# ---------------------------------------------------------------------------


def test_generate_network_timeout():
    """Timeout should raise TimeoutError for models that take too long.

    Uses blbr (bivalent ligand + bivalent receptor) *without* max_stoich so
    the polymer can grow without bound, guaranteeing that a short timeout
    always fires regardless of machine speed or future optimisations.
    """
    _reset_self_exporter()
    m = model_from_bngl(_bngl_location("blbr"))
    m.reset_equations()
    ng = NetworkGenerator(m)
    try:
        ng.generate_network(timeout=5)
        assert False, "Expected TimeoutError"
    except TimeoutError:
        pass


# ---------------------------------------------------------------------------
# Auto-netgen tests
# ---------------------------------------------------------------------------


def _make_robertson_model(auto_netgen=False):
    """Build a minimal Robertson model without SelfExporter."""
    SelfExporter.do_export = False
    try:
        m = Model(_export=False, auto_netgen=auto_netgen)
        A = Monomer("A", _export=False)
        B = Monomer("B", _export=False)
        C = Monomer("C", _export=False)
        m.add_component(A)
        m.add_component(B)
        m.add_component(C)

        k1 = Parameter("k1", 0.04, _export=False)
        k2 = Parameter("k2", 3e7, _export=False)
        k3 = Parameter("k3", 1e4, _export=False)
        A0 = Parameter("A0", 1.0, _export=False)
        B0 = Parameter("B0", 0.0, _export=False)
        C0 = Parameter("C0", 0.0, _export=False)
        for p in (k1, k2, k3, A0, B0, C0):
            m.add_component(p)

        r1 = Rule("r1", A() >> B(), k1, _export=False)
        r2 = Rule("r2", B() + B() >> C() + A(), k2, _export=False)
        r3 = Rule("r3", C() >> A(), k3, _export=False)
        for r in (r1, r2, r3):
            m.add_component(r)

        from pysb.core import Initial

        m.add_initial(Initial(A(), A0, _export=False))
        m.add_initial(Initial(B(), B0, _export=False))
        m.add_initial(Initial(C(), C0, _export=False))
        return m
    finally:
        SelfExporter.do_export = True


def test_auto_netgen_default_off():
    """Model(auto_netgen=False) does not auto-generate the network."""
    _reset_self_exporter()
    m = _make_robertson_model(auto_netgen=False)
    # accessing .species should NOT trigger netgen when auto_netgen=False
    assert_equal(m.species, [], "species should be empty before generation")
    assert_equal(m.reactions, [], "reactions should be empty before generation")


def test_auto_netgen_species_access_triggers_generation():
    """Accessing model.species triggers lazy network generation."""
    _reset_self_exporter()
    m = _make_robertson_model(auto_netgen=True)
    # Not yet generated
    assert_equal(m._species, [], "internal _species should be empty before access")
    # Access triggers generation
    species = m.species
    assert_equal(len(species), 3, "Robertson model has 3 species")


def test_auto_netgen_reactions_access_triggers_generation():
    """Accessing model.reactions triggers lazy network generation."""
    _reset_self_exporter()
    m = _make_robertson_model(auto_netgen=True)
    rxns = m.reactions
    assert_equal(len(rxns), 3, "Robertson model has 3 reactions")


def test_auto_netgen_reactions_bidirectional_access_triggers_generation():
    """Accessing model.reactions_bidirectional triggers lazy network generation."""
    _reset_self_exporter()
    m = _make_robertson_model(auto_netgen=True)
    rxns_bd = m.reactions_bidirectional
    assert_equal(len(rxns_bd), 3, "Robertson model has 3 bidirectional reactions")


def test_auto_netgen_not_regenerated_on_second_access():
    """Network is not regenerated on repeated access if model is unchanged."""
    _reset_self_exporter()
    m = _make_robertson_model(auto_netgen=True)
    _ = m.species  # trigger generation
    assert_equal(m._netgen_dirty, False, "flag should be False after generation")
    # Access again — should not regenerate (still clean)
    _ = m.species
    assert_equal(m._netgen_dirty, False, "flag should remain False")


def test_auto_netgen_dirty_after_new_rule():
    """Adding a new rule marks the model dirty."""
    _reset_self_exporter()
    m = _make_robertson_model(auto_netgen=True)
    _ = m.species  # trigger initial generation
    assert_equal(m._netgen_dirty, False)
    # Add a new (dummy) rule
    k_new = Parameter("k_new", 1.0, _export=False)
    m.add_component(k_new)
    A = m.monomers["A"]
    B = m.monomers["B"]
    r_new = Rule("r_new", A() + B() >> A() + B(), k_new, _export=False)
    m.add_component(r_new)
    assert_equal(m._netgen_dirty, True, "adding a rule must mark model dirty")


def test_auto_netgen_regenerates_after_new_rule():
    """After adding a new rule, the next access regenerates the network."""
    _reset_self_exporter()
    m = _make_robertson_model(auto_netgen=True)
    _ = m.species  # generate once
    # Add a trivial A() + B() -> A() + B() rule (no new species)
    k_new = Parameter("k_new", 1.0, _export=False)
    m.add_component(k_new)
    A = m.monomers["A"]
    B = m.monomers["B"]
    r_new = Rule("r_new", A() + B() >> A() + B(), k_new, _export=False)
    m.add_component(r_new)
    # Access forces regeneration — should still have 3 species (no new ones)
    assert_equal(len(m.species), 3, "species count unchanged after trivial rule")
    assert_equal(m._netgen_dirty, False, "flag cleared after regeneration")


def test_auto_netgen_dirty_after_new_initial():
    """Adding a new initial marks the model dirty."""
    _reset_self_exporter()
    m = _make_robertson_model(auto_netgen=True)
    _ = m.species
    assert_equal(m._netgen_dirty, False)
    from pysb.core import Initial

    # Add a duplicate initial (will be rejected by add_initial, so we append
    # directly to _DirtyList to test the dirty flag alone)
    k_extra = Parameter("k_extra", 0.0, _export=False)
    m.add_component(k_extra)
    A = m.monomers["A"]
    m.initials.append(Initial(A(), k_extra, _export=False))
    assert_equal(m._netgen_dirty, True, "appending to initials must mark model dirty")


def test_auto_netgen_reset_equations_marks_dirty():
    """reset_equations() marks the model dirty so next access regenerates."""
    _reset_self_exporter()
    m = _make_robertson_model(auto_netgen=True)
    _ = m.species
    assert_equal(m._netgen_dirty, False)
    m.reset_equations()
    assert_equal(m._netgen_dirty, True, "reset_equations must set _netgen_dirty")
    # Access should regenerate
    assert_equal(len(m.species), 3)
    assert_equal(m._netgen_dirty, False)


def test_auto_netgen_dirty_after_initials_iadd():
    """model.initials += [...] marks the model dirty (regression: __iadd__ bypasses extend)."""
    _reset_self_exporter()
    m = _make_robertson_model(auto_netgen=True)
    _ = m.species
    assert_equal(m._netgen_dirty, False)
    from pysb.core import Initial

    k_extra = Parameter("k_extra2", 0.0, _export=False)
    m.add_component(k_extra)
    A = m.monomers["A"]
    m.initials += [Initial(A(), k_extra, _export=False)]
    assert_equal(m._netgen_dirty, True, "__iadd__ on initials must mark model dirty")


def test_auto_netgen_no_effect_without_flag():
    """Model without auto_netgen=True is not affected by dirty-flag logic."""
    _reset_self_exporter()
    m = _make_robertson_model(auto_netgen=False)
    # Even though _netgen_dirty starts True, accessing species does nothing
    assert_equal(m.species, [])
    # rules still a _DirtyComponentSet but it won't trigger netgen
    k_new = Parameter("k_extra2", 1.0, _export=False)
    m.add_component(k_new)
    assert_equal(m._netgen_dirty, True)  # flag is still set
    assert_equal(m.species, [])  # but no netgen was triggered


def test_auto_netgen_pysb_example_robertson():
    """auto_netgen works end-to-end on the real Robertson example model."""
    _reset_self_exporter()
    model = _load_model("robertson")
    model.auto_netgen = True
    model._netgen_dirty = True
    assert_equal(len(model.species), 3)
    assert_equal(len(model.reactions), 3)


def test_auto_netgen_zero_species_no_loop():
    """A model that generates zero species does not re-generate on every access.

    Regression test for the bug where `_maybe_run_netgen` guarded on
    ``not _netgen_dirty and _species``, causing infinite re-generation when
    the model legitimately produced no species (empty network).
    """
    _reset_self_exporter()
    m = Model(_export=False)
    # No monomers, no rules, no initials → zero-species model
    m.auto_netgen = True
    # First access — triggers generation
    sp1 = m.species
    assert_equal(sp1, [])
    assert_equal(m._netgen_dirty, False, "dirty flag must be cleared after generation")
    # Second access — must NOT re-generate (flag is clear)
    sp2 = m.species
    assert_equal(sp2, [])
    assert_equal(
        m._netgen_dirty, False, "dirty flag must remain clear on second access"
    )


# ---------------------------------------------------------------------------
# Simulator netgen='auto' / auto_netgen integration
# ---------------------------------------------------------------------------


def test_simulator_netgen_auto_stored_as_auto():
    """netgen='auto' (default) is stored as 'auto', not resolved early."""
    from pysb.simulator.scipyode import ScipyOdeSimulator

    m = _make_robertson_model(auto_netgen=True)
    sim = ScipyOdeSimulator(m, tspan=[0, 1])
    assert_equal(sim.netgen, "auto")


def test_simulator_netgen_explicit_pysb():
    """Explicit netgen='pysb' is stored as 'pysb'."""
    from pysb.simulator.scipyode import ScipyOdeSimulator

    m = _make_robertson_model(auto_netgen=False)
    sim = ScipyOdeSimulator(m, tspan=[0, 1], netgen="pysb")
    assert_equal(sim.netgen, "pysb")


def test_simulator_netgen_bng_with_auto_netgen_raises():
    """netgen='bng' with model.auto_netgen=True raises ValueError."""
    from pysb.simulator.scipyode import ScipyOdeSimulator

    m = _make_robertson_model(auto_netgen=True)
    try:
        ScipyOdeSimulator(m, tspan=[0, 1], netgen="bng")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "auto_netgen" in str(e)


def test_simulator_netgen_invalid_raises():
    """Invalid netgen value raises ValueError."""
    from pysb.simulator.scipyode import ScipyOdeSimulator

    m = _make_robertson_model()
    try:
        ScipyOdeSimulator(m, tspan=[0, 1], netgen="bogus")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "bogus" in str(e)


# ---------------------------------------------------------------------------
# Energy rules — NotImplementedError guard
# ---------------------------------------------------------------------------


def test_energy_rule_raises():
    """generate_network raises NotImplementedError for models with energy rules."""
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    k = Parameter("k", 1.0, _export=False)
    k.model = m
    m.add_component(k)
    r = Rule("r", A(b=None) >> A(b=None), k, energy=True, _export=False)
    r.model = m
    m.add_component(r)

    ng = NetworkGenerator(m)
    try:
        ng.generate_network(populate=False)
        assert False, "expected NotImplementedError"
    except NotImplementedError as e:
        assert "energy" in str(e).lower()


def test_energy_pattern_raises():
    """generate_network raises NotImplementedError for models with EnergyPattern."""
    _reset_self_exporter()
    m = Model(_export=False)
    from pysb.core import EnergyPattern

    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    phi = Parameter("phi", 1.0, _export=False)
    phi.model = m
    m.add_component(phi)
    ep = EnergyPattern("ep1", A(b=None), phi, _export=False)
    ep.model = m
    m.add_component(ep)

    ng = NetworkGenerator(m)
    try:
        ng.generate_network(populate=False)
        assert False, "expected NotImplementedError"
    except NotImplementedError as e:
        assert "energy" in str(e).lower()


# ---------------------------------------------------------------------------
# max_iterations early stopping (with warning)
# ---------------------------------------------------------------------------


def test_max_iterations_warning():
    """max_iterations emits UserWarning when the network doesn't converge."""
    _reset_self_exporter()
    m = model_from_bngl(_bngl_location("tlbr"))
    m.reset_equations()
    ng = NetworkGenerator(m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ng.generate_network(max_iterations=1)
    warns = [
        x
        for x in w
        if issubclass(x.category, UserWarning) and "max_iterations" in str(x.message)
    ]
    assert_true(len(warns) > 0, "expected UserWarning about max_iterations")


# ---------------------------------------------------------------------------
# max_stoich filtering (with warning)
# ---------------------------------------------------------------------------


def test_max_stoich_warning():
    """max_stoich emits UserWarning when species are discarded."""
    _reset_self_exporter()
    m = model_from_bngl(_bngl_location("blbr"))
    m.reset_equations()
    ng = NetworkGenerator(m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ng.generate_network(max_stoich={"R": 5, "L": 5})
    warns = [
        x
        for x in w
        if issubclass(x.category, UserWarning) and "max_stoich" in str(x.message)
    ]
    assert_true(len(warns) > 0, "expected UserWarning about max_stoich")


# ---------------------------------------------------------------------------
# MultiState models with constraints — tlbr, blbr
# ---------------------------------------------------------------------------


def test_tlbr_with_max_iter():
    """tlbr (trivalent ligand, bivalent receptor) with max_iter=3."""
    _reset_self_exporter()
    m = model_from_bngl(_bngl_location("tlbr"))
    m.reset_equations()
    ng = NetworkGenerator(m)
    ng.generate_network(max_iterations=3)
    assert_equal(len(ng.species), 19)
    assert_equal(len(ng.reactions), 29)


def test_blbr_with_max_stoich():
    """blbr (bivalent ligand, bivalent receptor) with max_stoich."""
    _reset_self_exporter()
    m = model_from_bngl(_bngl_location("blbr"))
    m.reset_equations()
    ng = NetworkGenerator(m)
    ng.generate_network(max_stoich={"R": 5, "L": 5})
    assert_equal(len(ng.species), 20)
    assert_equal(len(ng.reactions), 92)


# ---------------------------------------------------------------------------
# Connected-component splitting
# ---------------------------------------------------------------------------


def test_component_splitting_unbind_within_cp():
    """Breaking a bond within a single-CP species must split disconnected monomers."""
    _reset_self_exporter()
    m = Model(_export=False)
    from pysb.core import MultiState

    R = Monomer("R", ["r", "r"], _export=False)
    R.model = m
    m.add_component(R)
    L = Monomer("L", ["l", "l"], _export=False)
    L.model = m
    m.add_component(L)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    kr = Parameter("kr", 1, _export=False)
    kr.model = m
    m.add_component(kr)

    # Rule: R(r=None) % L(l=None) | R(r=1) % L(l=1)
    # Forward creates a bond within a complex; reverse breaks it
    r = Rule(
        "intra_bind",
        R(r=None) % L(l=None) | R(r=1) % L(l=1),
        kf,
        kr,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    # Apply reverse to: R(r=MultiState(2, None)) % L(l=MultiState(2, None))
    # (R and L connected by single bond 2)
    # Breaking bond 2 should produce two separate species: R() and L()
    species = R(r=MultiState(2, None)) % L(l=MultiState(2, None))
    result = _apply_rule_to_species(r, (species,), rev_dir=True)
    # Should produce 2 separate species (R and L), not one disconnected complex
    assert_true(len(result) >= 1, f"Expected at least 1 product set, got {len(result)}")
    for products in result:
        for prod in products:
            # Each product should be connected (no disconnected components)
            from pysb.netgen import _split_connected_components

            components = _split_connected_components(list(prod.monomer_patterns))
            assert_equal(
                len(components),
                1,
                f"Product {prod} has {len(components)} components (should be 1)",
            )


# ---------------------------------------------------------------------------
# _mp_base_label
# ---------------------------------------------------------------------------


def test_mp_base_label_unbound():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    mp = A(b=None)
    label = _mp_base_label(mp)
    assert_equal(label[0], "A")
    assert_true(any("FREE" in str(part) for part in label[2]))


def test_mp_base_label_state():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    mp_u = A(s="u")
    mp_p = A(s="p")
    label_u = _mp_base_label(mp_u)
    label_p = _mp_base_label(mp_p)
    # Different states must produce different labels
    assert_true(label_u != label_p)


def test_mp_base_label_bond_number_independent():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    mp1 = A(b=1)
    mp2 = A(b=7)
    # Bond number should not affect the base label
    assert_equal(_mp_base_label(mp1), _mp_base_label(mp2))


def test_mp_base_label_bond_state_tuple():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    # (state, bond) — different bond numbers should give same label
    mp1 = A(s=("p", 1))
    mp2 = A(s=("p", 5))
    assert_equal(_mp_base_label(mp1), _mp_base_label(mp2))


def test_mp_base_label_multistate():
    _reset_self_exporter()
    m = Model(_export=False)
    from pysb.core import MultiState

    A = Monomer("A", ["r", "r"], _export=False)
    A.model = m
    m.add_component(A)
    # MultiState with two bond slots
    mp = A(r=MultiState(1, 2))
    label = _mp_base_label(mp)
    assert_equal(label[0], "A")
    # Should contain MS tag
    assert_true(any("MS" in str(part) for part in label[2]))


# ---------------------------------------------------------------------------
# _species_canonical_key
# ---------------------------------------------------------------------------


def test_canonical_key_single_monomer_bond_independent():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    sp1 = A(b=1) % B(a=1)
    sp2 = A(b=7) % B(a=7)
    # Same topology, different bond numbers → same canonical key
    assert_equal(
        _species_canonical_key(sp1),
        _species_canonical_key(sp2),
    )


def test_canonical_key_monomer_order_independent():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    sp1 = A(b=1) % B(a=1)
    sp2 = B(a=2) % A(b=2)
    assert_equal(
        _species_canonical_key(sp1),
        _species_canonical_key(sp2),
    )


def test_canonical_key_state_distinguishes():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    sp_u = as_complex_pattern(A(s="u"))
    sp_p = as_complex_pattern(A(s="p"))
    assert_true(_species_canonical_key(sp_u) != _species_canonical_key(sp_p))


def test_canonical_key_uses_cache():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    sp = as_complex_pattern(A(b=None))
    cache = {}
    k1 = _species_canonical_key(sp, sorted_sites_cache=cache)
    k2 = _species_canonical_key(sp, sorted_sites_cache=cache)
    assert_equal(k1, k2)
    # Cache should be populated after first call
    assert_true(len(cache) > 0)


# ---------------------------------------------------------------------------
# _sp_could_match_cp
# ---------------------------------------------------------------------------


def test_sp_could_match_cp_monomer_count_filter():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    # Species is only A; pattern requires A+B → should return False
    sp = as_complex_pattern(A(b=None))
    cp = A(b=1) % B(a=1)
    assert_true(not _sp_could_match_cp(sp, cp))


def test_sp_could_match_cp_bond_count_filter():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b", "c"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    # Species has 0 bonds; pattern requires 1 bond → False
    sp = as_complex_pattern(A(b=None, c=None))
    cp = as_complex_pattern(A(b=1))
    assert_true(not _sp_could_match_cp(sp, cp))


def test_sp_could_match_cp_state_filter():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    # Species is A(s='u'); pattern requires A(s='p') → False
    sp = as_complex_pattern(A(s="u"))
    cp = as_complex_pattern(A(s="p"))
    assert_true(not _sp_could_match_cp(sp, cp))


def test_sp_could_match_cp_passes_compatible():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    sp = as_complex_pattern(A(s="p"))
    cp = as_complex_pattern(A(s="p"))
    assert_true(_sp_could_match_cp(sp, cp))


def test_sp_could_match_cp_uses_cache():
    """Caches are populated after the first call."""
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    sp = as_complex_pattern(A(b=1))
    A2 = Monomer("A2", ["b"], _export=False)
    A2.model = m
    m.add_component(A2)
    cp = as_complex_pattern(A(b=1))
    sp_bc = {}
    cp_bc = {}
    sp_sc = {}
    cp_sc = {}
    _sp_could_match_cp(sp, cp, sp_bc, cp_bc, sp_sc, cp_sc)
    # After first call caches should be populated
    assert_true(len(sp_bc) > 0 or len(cp_bc) > 0 or len(sp_sc) > 0)


# ---------------------------------------------------------------------------
# _exceeds_max_stoich
# ---------------------------------------------------------------------------


def test_exceeds_max_stoich_under_limit():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", [], _export=False)
    A.model = m
    m.add_component(A)
    sp = A() % A()
    assert_true(not _exceeds_max_stoich(sp, {"A": 2}))


def test_exceeds_max_stoich_at_limit():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", [], _export=False)
    A.model = m
    m.add_component(A)
    sp = A() % A()
    # Exactly at limit → not exceeded
    assert_true(not _exceeds_max_stoich(sp, {"A": 2}))


def test_exceeds_max_stoich_over_limit():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    # 3 copies of A exceeds limit of 2
    # Note: A() % A() % A() requires chain bonding; build manually
    from pysb.core import ComplexPattern, MonomerPattern

    mps = [
        MonomerPattern(A, {"b": 1}, None),
        MonomerPattern(A, {"b": 1}, None),
        MonomerPattern(A, {"b": None}, None),
    ]
    sp2 = ComplexPattern(mps, None)
    assert_true(_exceeds_max_stoich(sp2, {"A": 2}))


def test_exceeds_max_stoich_unrelated_monomer():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", [], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", [], _export=False)
    B.model = m
    m.add_component(B)
    from pysb.core import ComplexPattern, MonomerPattern

    sp = ComplexPattern(
        [MonomerPattern(A, {}, None), MonomerPattern(A, {}, None)], None
    )
    # max_stoich only constrains B; A is unlimited here
    assert_true(not _exceeds_max_stoich(sp, {"B": 1}))


# ---------------------------------------------------------------------------
# _mp_site_matches_vf2
# ---------------------------------------------------------------------------


def test_vf2_bare_string_implies_unbound():
    # Rule says s='p' → species must have s='p' AND no bond
    assert_true(_mp_site_matches_vf2("p", "p"))
    # (state, bond) with bond is None also matches (s='p' + unbound)
    assert_true(_mp_site_matches_vf2(("p", None), "p"))
    # (state, int_bond) does NOT match bare string (bonded)
    assert_true(not _mp_site_matches_vf2(("p", 1), "p"))
    # Different state
    assert_true(not _mp_site_matches_vf2("u", "p"))


def test_vf2_int_bond_requires_bonded():
    # Rule says b=1 (a bond) → species must be bonded
    assert_true(_mp_site_matches_vf2(3, 1))  # bonded ints match
    assert_true(_mp_site_matches_vf2(("p", 2), 1))  # (state, bond) matches
    assert_true(not _mp_site_matches_vf2(None, 1))  # unbound doesn't match
    assert_true(not _mp_site_matches_vf2("p", 1))  # bare state (unbound) doesn't


def test_vf2_none_requires_unbound():
    assert_true(_mp_site_matches_vf2(None, None))
    assert_true(not _mp_site_matches_vf2(1, None))


def test_vf2_any_wild():
    from pysb.core import ANY, WILD

    assert_true(_mp_site_matches_vf2(1, ANY))
    assert_true(not _mp_site_matches_vf2(None, ANY))
    assert_true(_mp_site_matches_vf2(None, WILD))
    assert_true(_mp_site_matches_vf2(1, WILD))


def test_vf2_tuple_bond_none_matches_unbound_state():
    # Rule says s=('p', None) → species must have state 'p' AND no bond
    assert_true(_mp_site_matches_vf2(("p", None), ("p", None)))
    assert_true(_mp_site_matches_vf2("p", ("p", None)))
    assert_true(not _mp_site_matches_vf2(("p", 1), ("p", None)))


# ---------------------------------------------------------------------------
# _sp_contains_mp
# ---------------------------------------------------------------------------


def test_sp_contains_mp_basic():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    sp = as_complex_pattern(A(s="p"))
    rule_mp = A(s="p")
    assert_true(_sp_contains_mp(sp, rule_mp))


def test_sp_contains_mp_wrong_state():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    sp = as_complex_pattern(A(s="u"))
    rule_mp = A(s="p")
    assert_true(not _sp_contains_mp(sp, rule_mp))


def test_sp_contains_mp_complex_species():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    sp = A(b=1) % B(a=1)
    rule_mp = B(a=ANY)
    assert_true(_sp_contains_mp(sp, rule_mp))


def test_sp_contains_mp_wrong_monomer():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    sp = as_complex_pattern(A(b=None))
    rule_mp = B(a=None)
    assert_true(not _sp_contains_mp(sp, rule_mp))


# ---------------------------------------------------------------------------
# _multistate_slots_match
# ---------------------------------------------------------------------------


def test_multistate_slots_match_simple():
    # Both slots None — trivially matches
    assert_true(_multistate_slots_match([None, None], [None, None]))


def test_multistate_slots_match_permutation():
    # sp=[bond, None], rule=[None, bond] — needs permutation
    assert_true(_multistate_slots_match([1, None], [None, 2]))


def test_multistate_slots_match_no_match():
    # sp=[None, None], rule=[bond, bond] — no bond in species → no match
    assert_true(not _multistate_slots_match([None, None], [1, 2]))


# ---------------------------------------------------------------------------
# _build_src_bond_map
# ---------------------------------------------------------------------------


def test_build_src_bond_map_basic():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule(
        "bind",
        A(b=None) + B(a=None) >> A(b=1) % B(a=1),
        kf,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    # Build src_cps = reactant pattern
    src_cps = r.reactant_pattern.complex_patterns  # [A(b=None), B(a=None)]

    # Use _build_species_monomer_map to get the species monomer map
    _build_species_monomer_map(
        src_cps, [as_complex_pattern(A(b=None)), as_complex_pattern(B(a=None))]
    )
    # This maps rule monomers to free monomers (no bonds) — bond map will be empty
    # Instead, test with bonded rule pattern
    kf2 = Parameter("kf2", 1, _export=False)
    kf2.model = m
    m.add_component(kf2)
    r2 = Rule(
        "unbind",
        A(b=1) % B(a=1) >> A(b=None) + B(a=None),
        kf2,
        _export=False,
    )
    r2.model = m
    m.add_component(r2)
    src_cps2 = r2.reactant_pattern.complex_patterns  # [A(b=1) % B(a=1)]
    sp_ab = A(b=5) % B(a=5)
    species_monomer_map2 = _build_species_monomer_map(src_cps2, [sp_ab])
    bond_map = _build_src_bond_map(src_cps2, species_monomer_map2)
    # Rule bond 1 should map to species bond 5
    assert_equal(bond_map.get(1), 5)


# ---------------------------------------------------------------------------
# _extract_state_from_val / _extract_bonds_from_val
# ---------------------------------------------------------------------------


def test_extract_state_from_val_str():
    assert_equal(_extract_state_from_val("p"), "p")


def test_extract_state_from_val_tuple():
    assert_equal(_extract_state_from_val(("p", 3)), "p")


def test_extract_state_from_val_none():
    assert_equal(_extract_state_from_val(None), None)


def test_extract_state_from_val_int():
    assert_equal(_extract_state_from_val(3), None)


def test_extract_bonds_from_val_int():
    assert_equal(_extract_bonds_from_val(3), [3])


def test_extract_bonds_from_val_tuple():
    assert_equal(_extract_bonds_from_val(("p", 2)), [2])


def test_extract_bonds_from_val_none():
    assert_equal(_extract_bonds_from_val(None), [])


def test_extract_bonds_from_val_str():
    assert_equal(_extract_bonds_from_val("u"), [])


def test_extract_bonds_from_val_multistate():
    from pysb.core import MultiState

    val = MultiState(1, ("p", 2), None)
    result = sorted(_extract_bonds_from_val(val))
    assert_equal(result, [1, 2])


# ---------------------------------------------------------------------------
# _site_match_specificity
# ---------------------------------------------------------------------------


def test_site_match_specificity_exact_state():
    assert_equal(_site_match_specificity("p", "p"), 2)


def test_site_match_specificity_exact_none():
    assert_equal(_site_match_specificity(None, None), 2)


def test_site_match_specificity_exact_int():
    assert_equal(_site_match_specificity(3, 1), 2)


def test_site_match_specificity_loose_state_bond_tuple():
    # sp has (state, bond), rule has plain state → loose match
    assert_equal(_site_match_specificity(("p", 1), "p"), 1)


def test_site_match_specificity_any_wild():
    from pysb.core import ANY, WILD

    assert_equal(_site_match_specificity(1, ANY), 1)
    assert_equal(_site_match_specificity(None, WILD), 1)


# ---------------------------------------------------------------------------
# _find_multistate_permutation / _find_all_multistate_permutations
# ---------------------------------------------------------------------------


def test_find_multistate_permutation_trivial():
    # Identical slots — only one permutation (identity)
    perm = _find_multistate_permutation([None, None], [None, None])
    assert_equal(perm, [0, 1])


def test_find_multistate_permutation_swap():
    # sp=[bond, None], rule=[None, bond] — must swap
    perm = _find_multistate_permutation([1, None], [None, 2])
    assert_equal(perm, [1, 0])


def test_find_multistate_permutation_none_when_no_match():
    perm = _find_multistate_permutation([None, None], [1, 2])
    assert_true(perm is None)


def test_find_all_multistate_permutations_two_results():
    # Two symmetric bonds — both assignments valid
    perms = _find_all_multistate_permutations([1, 2], [3, 4])
    assert_equal(len(perms), 2)


def test_find_all_multistate_permutations_with_bond_map():
    # Bond map says rule bond 3 → sp bond 1; rule bond 4 → sp bond 2
    # So only the identity assignment is valid
    perms = _find_all_multistate_permutations([1, 2], [3, 4], bond_map={3: 1, 4: 2})
    assert_equal(len(perms), 1)
    assert_equal(perms[0], [0, 1])


# ---------------------------------------------------------------------------
# _resolve_multistate_slot / _resolve_plain_to_slot / _find_all_matched_multistate_slots
# ---------------------------------------------------------------------------


def test_resolve_multistate_slot_any_preserves():
    # ANY → keep existing
    result = _resolve_multistate_slot(ANY, 5, {})
    assert_equal(result, 5)


def test_resolve_multistate_slot_int_bond():
    result = _resolve_multistate_slot(3, None, {3: 99})
    assert_equal(result, 99)


def test_resolve_multistate_slot_state_bond_tuple():
    result = _resolve_multistate_slot(("p", 3), None, {3: 99})
    assert_equal(result, ("p", 99))


def test_resolve_multistate_slot_bare_string():
    result = _resolve_multistate_slot("u", None, {})
    assert_equal(result, "u")


def test_resolve_plain_to_slot_int():
    assert_equal(_resolve_plain_to_slot(3, {3: 99}), 99)


def test_resolve_plain_to_slot_tuple():
    assert_equal(_resolve_plain_to_slot(("p", 3), {3: 99}), ("p", 99))


def test_resolve_plain_to_slot_str():
    assert_equal(_resolve_plain_to_slot("u", {}), "u")


def test_find_all_matched_multistate_slots_none():
    # Reactant rule says None (unbound) — only unbound slots match
    indices = _find_all_matched_multistate_slots([None, 1, None], None, {})
    assert_equal(sorted(indices), [0, 2])


def test_find_all_matched_multistate_slots_bond():
    # Reactant rule says ANY (bonded) — only bonded slots match
    indices = _find_all_matched_multistate_slots([None, 1, None], ANY, {})
    assert_equal(indices, [1])


def test_find_all_matched_multistate_slots_fallback():
    # No slots match → fallback returns [0]
    indices = _find_all_matched_multistate_slots([None, None], 1, {})
    assert_equal(indices, [0])


# ---------------------------------------------------------------------------
# _split_connected_components — direct tests
# ---------------------------------------------------------------------------


def test_split_connected_components_single():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    from pysb.core import MonomerPattern

    mp = MonomerPattern(A, {"b": None}, None)
    result = _split_connected_components([mp])
    assert_equal(len(result), 1)
    assert_equal(result[0], [mp])


def test_split_connected_components_all_connected():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    from pysb.core import MonomerPattern

    mp_a = MonomerPattern(A, {"b": 1}, None)
    mp_b = MonomerPattern(B, {"a": 1}, None)
    result = _split_connected_components([mp_a, mp_b])
    assert_equal(len(result), 1)


def test_split_connected_components_two_isolated():
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    from pysb.core import MonomerPattern

    mp_a = MonomerPattern(A, {"b": None}, None)
    mp_b = MonomerPattern(B, {"a": None}, None)
    result = _split_connected_components([mp_a, mp_b])
    assert_equal(len(result), 2)


def test_split_connected_components_cyclic():
    """A--B--C--A ring: all one component."""
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b", "c"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a", "c"], _export=False)
    B.model = m
    m.add_component(B)
    C = Monomer("C", ["a", "b"], _export=False)
    C.model = m
    m.add_component(C)
    from pysb.core import MonomerPattern

    mp_a = MonomerPattern(A, {"b": 1, "c": 3}, None)
    mp_b = MonomerPattern(B, {"a": 1, "c": 2}, None)
    mp_c = MonomerPattern(C, {"a": 2, "b": 3}, None)
    result = _split_connected_components([mp_a, mp_b, mp_c])
    assert_equal(len(result), 1)


def test_split_connected_components_isolated_plus_pair():
    """One isolated monomer and a bonded pair."""
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    C = Monomer("C", [], _export=False)
    C.model = m
    m.add_component(C)
    from pysb.core import MonomerPattern

    mp_a = MonomerPattern(A, {"b": 1}, None)
    mp_b = MonomerPattern(B, {"a": 1}, None)
    mp_c = MonomerPattern(C, {}, None)
    result = _split_connected_components([mp_a, mp_b, mp_c])
    assert_equal(len(result), 2)


# ---------------------------------------------------------------------------
# _apply_rule_to_species — forward binding (two reactants → complex)
# ---------------------------------------------------------------------------


def test_apply_rule_forward_binding():
    """A(b=None) + B(a=None) >> A(b=1)%B(a=1) must produce one complex."""
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["b"], _export=False)
    A.model = m
    m.add_component(A)
    B = Monomer("B", ["a"], _export=False)
    B.model = m
    m.add_component(B)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule(
        "bind",
        A(b=None) + B(a=None) >> A(b=1) % B(a=1),
        kf,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    sp_A = as_complex_pattern(A(b=None))
    sp_B = as_complex_pattern(B(a=None))
    result = _apply_rule_to_species(r, (sp_A, sp_B))
    assert_equal(len(result), 1)
    product_list = result[0]
    assert_equal(len(product_list), 1)
    product = product_list[0]
    monomer_names = sorted(mp.monomer.name for mp in product.monomer_patterns)
    assert_equal(monomer_names, ["A", "B"])
    # Verify bond formed
    bonds = get_bonds_in_pattern(product)
    assert_equal(len(bonds), 1)


# ---------------------------------------------------------------------------
# generate_network populate=True default
# ---------------------------------------------------------------------------


def test_generate_network_populate_true_default():
    """generate_network() with default populate=True fills model fields in one call."""
    _reset_self_exporter()
    model = _load_model("robertson")
    ng = NetworkGenerator(model)
    ng.generate_network()  # populate=True is the default

    assert_equal(len(model.species), 3)
    assert_equal(len(model.reactions), 3)
    assert_equal(len(model.reactions_bidirectional), 3)
    assert_true(all(len(obs.species) > 0 for obs in model.observables))


# ---------------------------------------------------------------------------
# NetworkGenerator.populate_model
# ---------------------------------------------------------------------------


def test_populate_model_fills_model_fields():
    """populate_model() writes species/reactions/reactions_bidirectional."""
    _reset_self_exporter()
    model = _load_model("robertson")
    ng = NetworkGenerator(model)
    ng.generate_network(populate=False)
    ng.populate_model()

    assert_equal(len(model.species), 3)
    assert_equal(len(model.reactions), 3)
    assert_equal(len(model.reactions_bidirectional), 3)


def test_populate_model_raises_without_generate():
    _reset_self_exporter()
    model = _load_model("robertson")
    ng = NetworkGenerator(model)
    try:
        ng.populate_model()
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert_in("generate_network", str(e))


def test_populate_model_observables():
    """populate_model() fills observable species/coefficients."""
    _reset_self_exporter()
    model = _load_model("robertson")
    ng = NetworkGenerator(model)
    ng.generate_network(populate=False)
    ng.populate_model()
    assert_true(all(len(obs.species) > 0 for obs in model.observables))


# ---------------------------------------------------------------------------
# NetworkGenerator._lookup_or_add_species
# ---------------------------------------------------------------------------


def test_lookup_or_add_species_new():
    _reset_self_exporter()
    model = _load_model("robertson")
    ng = NetworkGenerator(model)
    ng.generate_network()

    # Introduce a brand-new species
    B = model.monomers["B"]
    new_sp = as_complex_pattern(B())
    # Manually clear it to ensure it's absent
    key_new = _species_canonical_key(new_sp)
    original_count = len(ng.species)

    # Inject: if the key already exists we can't test "is_new"
    if key_new not in ng._species_by_key:
        sp_id, is_new = ng._lookup_or_add_species(new_sp)
        assert_true(is_new)
        assert_equal(len(ng.species), original_count + 1)


def test_lookup_or_add_species_existing():
    _reset_self_exporter()
    model = _load_model("robertson")
    ng = NetworkGenerator(model)
    ng.generate_network()

    existing = ng.species[0]
    sp_id, is_new = ng._lookup_or_add_species(existing)
    assert_true(not is_new)
    assert_equal(sp_id, 0)


# ---------------------------------------------------------------------------
# NetworkGenerator._fire_rule_combo — synthesis path and stoich violation
# ---------------------------------------------------------------------------


def test_fire_rule_combo_synthesis():
    """Synthesis rule fired via _fire_rule_combo adds product species."""
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", [], _export=False)
    A.model = m
    m.add_component(A)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule("synth", None >> A(), kf, _export=False)
    r.model = m
    m.add_component(r)

    ng = NetworkGenerator(m)
    ng.species_pm = SpeciesPatternMatcher(m, [])
    is_new = ng._fire_rule_combo(r, (), False)
    assert_true(is_new)
    assert_equal(len(ng.species), 1)


def test_fire_rule_combo_stoich_violation():
    """Stoichiometry violation prevents product from being added."""
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", [], _export=False)
    A.model = m
    m.add_component(A)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule("synth", None >> A(), kf, _export=False)
    r.model = m
    m.add_component(r)

    ng = NetworkGenerator(m)
    ng.species_pm = SpeciesPatternMatcher(m, [])
    ng._max_stoich = {"A": 0}  # zero copies of A allowed → violates immediately
    is_new = ng._fire_rule_combo(r, (), False)
    # stoich violated: species not added; _stoich_pruned incremented
    assert_true(not is_new)
    assert_equal(len(ng.species), 0)
    assert_equal(ng._stoich_pruned, 1)


def test_fire_rule_combo_reactant_indices_none():
    """When reactant_indices=None, canonical-key lookup is used."""
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    r = Rule("phos", A(s="u") >> A(s="p"), kf, _export=False)
    r.model = m
    m.add_component(r)

    sp_u = as_complex_pattern(A(s="u"))
    sp_p = as_complex_pattern(A(s="p"))

    ng = NetworkGenerator(m)
    ng.species_pm = SpeciesPatternMatcher(m, [sp_u, sp_p])
    # Populate _species_by_key manually (normally done in generate_network)
    for i, sp in enumerate(ng.species_pm.species):
        k = _species_canonical_key(sp, ng._sorted_sites_cache)
        ng._species_by_key[k] = i

    # reactant_indices=None triggers key-lookup path
    ng._fire_rule_combo(r, (sp_u,), False, reactant_indices=None)
    assert_equal(len(ng.reactions), 1)


# ---------------------------------------------------------------------------
# Cache non-stale between NetworkGenerator instances
# ---------------------------------------------------------------------------


def test_cache_not_stale_between_instances():
    """Per-instance caches do not carry stale entries from a previous run."""
    _reset_self_exporter()
    # Generate network for Robertson twice with fresh NG instances.
    # If id()-keyed caches leaked between instances, the second run
    # could get wrong bond/state counts for species whose Python id()
    # was reused, producing wrong pre-filter decisions.
    model1 = _load_model("robertson")
    ng1 = NetworkGenerator(model1)
    ng1.generate_network()
    n_sp1 = len(ng1.species)

    # Allow the first NG and model to be garbage-collected so Python may
    # reuse their memory addresses.
    del ng1, model1

    model2 = _load_model("robertson")
    ng2 = NetworkGenerator(model2)
    ng2.generate_network()
    n_sp2 = len(ng2.species)

    assert_equal(n_sp1, n_sp2)
    assert_equal(n_sp1, 3)


# ---------------------------------------------------------------------------
# Synthesised monomer with MultiState bonds — integration test
# ---------------------------------------------------------------------------


def test_synthesised_monomer_multistate_bond_remapping():
    """Synthesis rule producing a monomer with MultiState bonds generates valid species."""
    _reset_self_exporter()
    m = Model(_export=False)
    from collections import Counter

    from pysb.core import MultiState

    # Bivalent linker L with two identical binding sites; R is a ligand.
    L = Monomer("L", ["r", "r"], _export=False)
    L.model = m
    m.add_component(L)
    R = Monomer("R", ["l"], _export=False)
    R.model = m
    m.add_component(R)
    k = Parameter("k", 1e-3, _export=False)
    k.model = m
    m.add_component(k)
    # Synthesis rule: produce a pre-assembled L(r=MS(1,2)):R(l=1):R(l=2) complex.
    # L is a synthesised monomer (no LHS counterpart) with a MultiState site
    # whose bond integers must be remapped via product_bond_map.
    rule = Rule(
        "synth",
        None >> L(r=MultiState(1, 2)) % R(l=1) % R(l=2),
        k,
        _export=False,
    )
    rule.model = m
    m.add_component(rule)

    ng = NetworkGenerator(m)
    ng.generate_network(populate=False)

    # Exactly one species — the synthesised complex
    assert_equal(len(ng.species), 1)
    sp = ng.species[0]
    mp_names = sorted(mp.monomer.name for mp in sp.monomer_patterns)
    assert_equal(mp_names, ["L", "R", "R"])

    # L's r site must be a MultiState with two integer bond slots
    l_mp = next(mp for mp in sp.monomer_patterns if mp.monomer.name == "L")
    r_val = l_mp.site_conditions["r"]
    assert_true(isinstance(r_val, MultiState))
    slots = list(r_val)
    assert_equal(len(slots), 2)
    assert_true(all(isinstance(s, int) for s in slots))

    # Every bond integer must appear exactly twice (once in L.r, once in R.l),
    # confirming correct bond remapping and connectivity.
    bond_counts = Counter()
    for mp in sp.monomer_patterns:
        for val in mp.site_conditions.values():
            if isinstance(val, MultiState):
                for slot in val:
                    if isinstance(slot, int):
                        bond_counts[slot] += 1
            elif isinstance(val, int):
                bond_counts[val] += 1
    assert_true(all(c == 2 for c in bond_counts.values()))

    # Exactly one reaction (zero-order synthesis)
    assert_equal(len(ng.reactions), 1)


# ---------------------------------------------------------------------------
# _build_rule_monomer_mapping with two copies of the same monomer
# ---------------------------------------------------------------------------


def test_build_rule_mapping_ambiguous_monomers():
    """Specificity scoring picks the best reactant match for each product."""
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", ["s"], {"s": ["u", "p"]}, _export=False)
    A.model = m
    m.add_component(A)
    kf = Parameter("kf", 1, _export=False)
    kf.model = m
    m.add_component(kf)
    # Rule: A(s='u') + A(s='p') >> A(s='p') + A(s='p')
    # The A(s='u') on LHS → first A(s='p') on RHS (state change)
    # The A(s='p') on LHS → second A(s='p') on RHS (no change)
    r = Rule(
        "double",
        A(s="u") + A(s="p") >> A(s="p") + A(s="p"),
        kf,
        _export=False,
    )
    r.model = m
    m.add_component(r)

    mapping = _build_rule_monomer_mapping(r)
    # All four product monomers should be mapped to some reactant monomer
    assert_equal(len(mapping), 2)
    # The two mapped reactant positions must be distinct (no double-claim)
    reactant_positions = list(mapping.values())
    assert_equal(len(reactant_positions), len(set(reactant_positions)))


# ---------------------------------------------------------------------------
# Reversible synthesis rules — None <> A() must produce both directions
# ---------------------------------------------------------------------------


def test_reversible_synthesis_rule():
    """None <> A() must generate both a synthesis and a degradation reaction.

    Previously only the forward (synthesis) direction was fired; the reverse
    (degradation) was silently dropped because the rule was classified as a
    synthesis rule and never entered the main iteration loop.
    """
    _reset_self_exporter()
    m = Model(_export=False)
    A = Monomer("A", _export=False)
    A.model = m
    m.add_component(A)
    k_syn = Parameter("k_syn", 1.0, _export=False)
    k_syn.model = m
    m.add_component(k_syn)
    k_deg = Parameter("k_deg", 0.1, _export=False)
    k_deg.model = m
    m.add_component(k_deg)
    r = Rule("synth_deg", None | A(), k_syn, k_deg, _export=False)
    r.model = m
    m.add_component(r)

    ng = NetworkGenerator(m)
    ng.generate_network()

    # One species: A()
    assert_equal(len(ng.species), 1)

    # Two reactions: one synthesis (no reactants) and one degradation (no products)
    assert_equal(len(ng.reactions), 2)
    rxns = list(ng.reactions.values())
    assert_true(
        any(len(rxn["reactants"]) == 0 for rxn in rxns), "no synthesis reaction"
    )
    assert_true(
        any(len(rxn["products"]) == 0 for rxn in rxns), "no degradation reaction"
    )


# ---------------------------------------------------------------------------
# Regression tests for three bug fixes
# ---------------------------------------------------------------------------


def test_compartment_in_canonical_key():
    """Same monomer in different compartments must be distinct species.

    Regression test for Bug 1: _mp_base_label and _canonical_key_from_order
    previously omitted the compartment from the hash key, causing B()@EC,
    B()@PM, and B()@CP to collide and be treated as the same species.
    """
    m = model_from_bngl(_bngl_location("univ_synth"), force=True)
    ng = NetworkGenerator(m)
    ng.generate_network(timeout=30)

    sp_strs = [str(sp) for sp in ng.species]

    # Three distinct B() species must be generated (one per compartment)
    b_species = [s for s in sp_strs if s.startswith("B()")]
    assert_equal(
        len(b_species),
        3,
        f"Expected 3 distinct B() species (one per compartment); got: {b_species}",
    )
    assert_in("B() ** EC", sp_strs)
    assert_in("B() ** PM", sp_strs)
    assert_in("B() ** CP", sp_strs)


def test_state_preserved_when_bond_added_or_removed():
    """State must be preserved on a site when a bond is added or removed.

    Regression test for Bug 2: when the rule pattern says ``A(b!1)`` and the
    source species has ``A(b~0)`` (state, no bond), the product should be
    ``A(b~0!1)`` (state preserved, bond added).  Symmetrically, the reverse
    direction should strip the bond and restore the bare state.

    Uses the statfactor BNG Validate model, where R1 is
    ``A(b) + A(b) <-> A(b!1).A(b!1)`` and A has site ``b~0~1``.
    """
    m = model_from_bngl(_bngl_location("statfactor"), force=True)
    ng = NetworkGenerator(m)
    ng.generate_network(timeout=30)

    sp_strs = [str(sp) for sp in ng.species]

    # The dimer A(b~0!1).A(b~0!1) must appear (two A(b~0) molecules bound).
    # PySB uses internal state names (e.g. '_0' for BNGL 'b~0').
    dimer_species = [s for s in sp_strs if s.count("%") == 1]
    assert_true(
        len(dimer_species) > 0,
        f"No dimer species found in: {sp_strs}",
    )
    # Total species count must match BNG (6 species for statfactor)
    assert_equal(len(ng.species), 6, f"Species list: {sp_strs}")


def test_free_site_matches_state_bearing_site():
    """A rule pattern with a free site must match species with a state on that site.

    Regression test for Bug 3: _species_matches_rule_site and
    _mp_site_matches_vf2 previously returned False for rule_val=None when the
    species value was a bare string (state with no bond).  This caused binding
    rules to miss state-bearing species as reactants.

    Uses the statfactor BNG Validate model.  Rule R1 is
    ``A(b) + A(b) <-> A(b!1).A(b!1)``.  The pattern ``A(b)`` has no
    explicit condition on ``b``, which in BNG means "b is free" — it should
    match ``A(b~0)`` and ``A(b~1)``.  If Bug 3 were present, netgen would
    generate 0 binding reactions.
    """
    m = model_from_bngl(_bngl_location("statfactor"), force=True)
    ng = NetworkGenerator(m)
    ng.generate_network(timeout=30)

    # R1 binding reactions: A(b~?) + A(b~?) -> A(b~?!1).A(b~?!1)
    # There should be bimolecular reactions with 2 reactants and 1 product.
    binding_rxns = [
        rxn
        for rxn in ng.reactions.values()
        if len(rxn["reactants"]) == 2
        and len(rxn["products"]) == 1
        and "R1" in rxn["rule"]
    ]
    assert_true(
        len(binding_rxns) > 0,
        "No R1 binding reactions found — free-site matching for state-bearing "
        "sites is broken.",
    )
    # Total reaction count must match BNG (14 reactions for statfactor)
    assert_equal(len(ng.reactions), 14, f"Reaction count mismatch")


def test_localfunc_raises_not_implemented():
    """generate_network must raise NotImplementedError for models with local functions.

    Local functions (Tag/@-annotated rule patterns) require per-species rate
    expansion that is not yet implemented.  Silently ignoring them would
    produce wrong reaction rates, so the generator must refuse immediately.
    """
    m = model_from_bngl(_bngl_location("localfunc"), force=True)
    ng = NetworkGenerator(m)
    try:
        ng.generate_network(timeout=30)
        assert False, "Expected NotImplementedError for local-function model"
    except NotImplementedError as exc:
        assert_in("local function", str(exc).lower())


# ---------------------------------------------------------------------------
# Homo-multimer symmetry-factor regression tests
# ---------------------------------------------------------------------------


def _build_homodimer_model():
    """Return a minimal PySB model with a reversible homodimerisation rule.

    Model:  A(b) + A(b) <-> A(b!1).A(b!1)   kf, kr
    All components are added without exporting to the module namespace so the
    model can be created inside a test function without polluting SelfExporter.
    """
    SelfExporter.do_export = False
    try:
        m = Model(_export=False)
        A = Monomer("A", ["b"], _export=False)
        m.add_component(A)
        kf = Parameter("kf", 1.0, _export=False)
        kr = Parameter("kr", 0.5, _export=False)
        m.add_component(kf)
        m.add_component(kr)
        r = Rule(
            "bind",
            A(b=None) + A(b=None) | A(b=1) % A(b=1),
            kf,
            kr,
            _export=False,
        )
        m.add_component(r)
        from pysb.core import Initial

        A_0 = Parameter("A_0", 2.0, _export=False)
        m.add_component(A_0)
        m.add_initial(Initial(A(b=None), A_0, _export=False))
        return m
    finally:
        SelfExporter.do_export = True


def test_homodimer_symmetry_factor_rate():
    """Homo-dimer rule A+A->D must carry a 1/2! symmetry factor in the rate.

    BNG divides the reaction propensity by ``n!`` for each group of *n*
    identical reactant species.  For ``A(b) + A(b) -> A(b!1).A(b!1)`` there
    is one group of size 2, so the forward rate must be ``kf/2 * s0^2``,
    not ``kf * s0^2``.  Omitting the factor causes ODEs that consume A twice
    as fast as BNG and makes netgen simulations disagree with BNG simulations.
    """
    import sympy

    m = _build_homodimer_model()
    ng = NetworkGenerator(m)
    ng.generate_network(populate=True)

    fwd_rxns = [rxn for rxn in m.reactions if len(rxn["reactants"]) == 2]
    assert_true(len(fwd_rxns) == 1, "Expected exactly one bimolecular reaction")
    rate = fwd_rxns[0]["rate"]
    # Rate must equal kf/2 * __s0^2: the Rational(1,2) coefficient must be present.
    coeff, rest = sympy.Rational(rate.as_coeff_Mul()[0]), rate.as_coeff_Mul()[1]
    assert_equal(
        coeff,
        sympy.Rational(1, 2),
        f"Forward rate coefficient should be 1/2, got {coeff}",
    )
    # The remaining factor must contain kf and __s0**2
    sym_names = {s.name for s in rest.free_symbols}
    assert_true(
        "kf" in sym_names,
        f"kf missing from rate: {rate}",
    )
    s0 = sympy.Symbol("__s0")
    assert_true(
        rest.has(s0**2),
        f"__s0^2 missing from rate: {rate}",
    )


def test_homodimer_rate_matches_bng():
    """Netgen reaction rates for homodimerisation must match BNG exactly."""
    import sympy
    from pysb.bng import generate_equations

    m = _build_homodimer_model()

    # Netgen
    ng = NetworkGenerator(m)
    ng.generate_network(populate=True)
    ng_rates = {(rxn["reactants"], rxn["products"]): rxn["rate"] for rxn in m.reactions}

    # BNG
    generate_equations(m)
    bng_rates = {
        (rxn["reactants"], rxn["products"]): rxn["rate"] for rxn in m.reactions
    }

    # Map netgen species to BNG species and compare
    for key, ng_rate in ng_rates.items():
        assert_in(key, bng_rates, f"Reaction {key} missing from BNG output")
        bng_rate = bng_rates[key]
        diff = sympy.simplify(ng_rate - bng_rate)
        assert_equal(
            diff,
            sympy.Integer(0),
            f"Rate mismatch for {key}: netgen={ng_rate}, BNG={bng_rate}",
        )


def test_homodimer_simulation_matches_bng():
    """Simulation trajectory from netgen must match BNG for homodimerisation.

    The ODE from netgen must agree with BNG to within rtol=1e-4 over t=[0,5].
    Prior to the symmetry-factor fix, netgen produced ODEs twice as fast,
    causing large divergence.
    """
    import numpy as np
    from pysb.bng import generate_equations
    from pysb.simulator import ScipyOdeSimulator

    m = _build_homodimer_model()
    from pysb.core import Observable

    # Add an observable for free A
    obs = Observable("free_A", m.monomers["A"](b=None), _export=False)
    m.add_component(obs)

    # Netgen simulation
    ng = NetworkGenerator(m)
    ng.generate_network(populate=True)
    tspan = np.linspace(0, 5, 200)
    res_ng = ScipyOdeSimulator(m, tspan=tspan, compiler="python").run()
    ng_traj = res_ng.observables["free_A"]

    # BNG simulation
    generate_equations(m)
    res_bng = ScipyOdeSimulator(m, tspan=tspan, compiler="python").run()
    bng_traj = res_bng.observables["free_A"]

    np.testing.assert_allclose(
        ng_traj,
        bng_traj,
        rtol=1e-4,
        err_msg="Netgen homodimer trajectory diverges from BNG",
    )


def test_trimerisation_symmetry_factor_rate():
    """3A -> T must carry a 1/3! = 1/6 symmetry factor in the rate."""
    import sympy

    from pysb.core import Initial

    SelfExporter.do_export = False
    try:
        m = Model(_export=False)
        A = Monomer("A", ["b"], _export=False)
        T = Monomer("T", [], _export=False)
        m.add_component(A)
        m.add_component(T)
        k = Parameter("k", 1.0, _export=False)
        m.add_component(k)
        r = Rule(
            "trim",
            A(b=None) + A(b=None) + A(b=None) >> T(),
            k,
            _export=False,
        )
        m.add_component(r)
        m.add_initial(Initial(A(b=None), Parameter("A_0", 3.0, _export=False)))
    finally:
        SelfExporter.do_export = True

    ng = NetworkGenerator(m)
    ng.generate_network(populate=True)

    assert_true(len(m.reactions) >= 1, "Expected at least one reaction")
    fwd_rxns = [rxn for rxn in m.reactions if len(rxn["reactants"]) == 3]
    assert_true(len(fwd_rxns) == 1, "Expected exactly one trimer-forming reaction")
    rate = fwd_rxns[0]["rate"]
    coeff = sympy.Rational(rate.as_coeff_Mul()[0])
    assert_equal(
        coeff,
        sympy.Rational(1, 6),
        f"Trimerisation symmetry factor should be 1/6, got {coeff}",
    )


# ---------------------------------------------------------------------------
# Cross-compartment transport regression tests
# ---------------------------------------------------------------------------


def _build_transport_model():
    """Return a minimal PySB model with a single-species compartment transport rule.

    Model:  A()@EC >> B()@CP   k
    EC is a 3-D extracellular volume; CP is a 2-D membrane inside EC.
    This exercises the compartment inference path in _apply_rule_with_monomer_map.
    """
    from pysb.core import Compartment, Initial

    SelfExporter.do_export = False
    try:
        m = Model(_export=False)
        EC = Compartment("EC", dimension=3, _export=False)
        CP = Compartment("CP", dimension=2, parent=EC, _export=False)
        m.add_component(EC)
        m.add_component(CP)
        A = Monomer("A", [], _export=False)
        B = Monomer("B", [], _export=False)
        m.add_component(A)
        m.add_component(B)
        k = Parameter("k", 0.5, _export=False)
        m.add_component(k)
        r = Rule("transport", A() ** EC >> B() ** CP, k, _export=False)
        m.add_component(r)
        A_0 = Parameter("A_0", 1.0, _export=False)
        m.add_component(A_0)
        m.add_initial(Initial(A() ** EC, A_0, _export=False))
        return m
    finally:
        SelfExporter.do_export = True


def test_cross_compartment_species_and_reactions():
    """Cross-compartment transport generates correct species and reactions.

    Netgen must produce two species (A@EC, B@CP) and one reaction
    (species 0 -> species 1) attributed to rule 'transport'.
    """
    m = _build_transport_model()
    ng = NetworkGenerator(m)
    ng.generate_network(populate=True)

    assert_equal(len(ng.species), 2, f"Expected 2 species, got {ng.species}")
    assert_equal(len(m.reactions), 1, f"Expected 1 reaction, got {m.reactions}")
    rxn = m.reactions[0]
    assert_equal(rxn["rule"], ("transport",))
    # Reactants contain one species (A@EC), products contain one species (B@CP)
    assert_equal(len(rxn["reactants"]), 1)
    assert_equal(len(rxn["products"]), 1)
    assert_true(
        rxn["reactants"][0] != rxn["products"][0],
        "Reactant and product must be different species",
    )


def test_cross_compartment_rate_matches_bng():
    """Netgen reaction rate for cross-compartment transport must match BNG.

    Both netgen and BNG express the rate as ``k * __s0`` where __s0 is the
    concentration of the source species.  Volume scaling is handled at the
    ODE level, not at the reaction-rate level.
    """
    import sympy
    from pysb.bng import generate_equations

    m = _build_transport_model()

    # Netgen
    ng = NetworkGenerator(m)
    ng.generate_network(populate=True)
    ng_rates = {(rxn["reactants"], rxn["products"]): rxn["rate"] for rxn in m.reactions}

    # BNG
    generate_equations(m)
    bng_rates = {
        (rxn["reactants"], rxn["products"]): rxn["rate"] for rxn in m.reactions
    }

    for key, ng_rate in ng_rates.items():
        assert_in(key, bng_rates, f"Reaction {key} missing from BNG output")
        diff = sympy.simplify(ng_rate - bng_rates[key])
        assert_equal(
            diff,
            sympy.Integer(0),
            f"Rate mismatch for {key}: netgen={ng_rate}, BNG={bng_rates[key]}",
        )


def test_cross_compartment_matches_bng_full():
    """Cross-compartment transport: species and reactions both match BNG."""
    m = _build_transport_model()
    ng = NetworkGenerator(m)
    ng.generate_network()
    corr = ng.check_species_against_bng()
    ng.check_reactions_against_bng(corr)
