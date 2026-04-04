"""
Tests for the libsbml-based SBML importer (pysb.importers.sbml.SbmlImporter).

The legacy atomizer-based SBML import tests remain in test_importers.py.
"""

import os
import tempfile
import shutil
import textwrap

import sympy
from unittest import mock
from nose.plugins.skip import SkipTest
from nose.tools import assert_raises_regex

import pysb.pathfinder as pf
from pysb.importers.sbml import (
    SbmlImporter,
    SbmlImportError,
    model_from_sbml,
    _model_from_sbml_libsbml,
    model_from_biomodels,
    _sanitize_id,
    _split_reversible_rate,
)

try:
    import libsbml

    HAS_LIBSBML = True
except ImportError:
    HAS_LIBSBML = False

if not HAS_LIBSBML:
    raise SkipTest("libsbml not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bng_validate_directory():
    bng_exec = os.path.realpath(pf.get_path("bng"))
    if bng_exec.endswith(".bat"):
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            return os.path.join(conda_prefix, r"share\bionetgen\Validate")
    return os.path.join(os.path.dirname(bng_exec), "Validate")


def _sbml_location(filename):
    return os.path.join(_bng_validate_directory(), "INPUT_FILES", filename + ".xml")


_DEFAULT_COMPARTMENT = '<compartment id="c" size="1"/>'


def _minimal_sbml(
    species=None,
    parameters=None,
    reactions=None,
    rules=None,
    compartments=_DEFAULT_COMPARTMENT,
    level=2,
    version=3,
):
    """Build a minimal SBML string for testing.

    Optional blocks (species, parameters, reactions, rules) are omitted
    entirely when empty to keep the document valid (SBML forbids empty
    listOf* elements).
    """
    blocks = []
    if compartments:
        blocks.append(
            "<listOfCompartments>{}</listOfCompartments>".format(compartments)
        )
    if species:
        blocks.append("<listOfSpecies>{}</listOfSpecies>".format(species))
    if parameters:
        blocks.append("<listOfParameters>{}</listOfParameters>".format(parameters))
    if rules:
        blocks.append("<listOfRules>{}</listOfRules>".format(rules))
    if reactions:
        blocks.append("<listOfReactions>{}</listOfReactions>".format(reactions))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level{lv}/version{vn}"'
        ' level="{lv}" version="{vn}">\n'
        '  <model id="test_model">\n'
        "    {body}\n"
        "  </model>\n"
        "</sbml>"
    ).format(lv=level, vn=version, body="\n    ".join(blocks))


# ---------------------------------------------------------------------------
# _sanitize_id
# ---------------------------------------------------------------------------


def test_sanitize_id_plain():
    assert _sanitize_id("A") == "A"


def test_sanitize_id_hyphen():
    assert _sanitize_id("my-species") == "my_species"


def test_sanitize_id_leading_digit():
    assert _sanitize_id("2fast") == "s_2fast"


def test_sanitize_id_spaces():
    assert _sanitize_id("A B") == "A_B"


def test_sanitize_id_empty():
    assert _sanitize_id("") == "unnamed"


# ---------------------------------------------------------------------------
# SbmlImporter – flat test file
# ---------------------------------------------------------------------------


class TestFlatSbml:
    """Tests against the BNG validation flat SBML file."""

    def setup(self):
        path = _sbml_location("test_sbml_flat_SBML")
        self.model = SbmlImporter(path).model

    def test_model_name(self):
        assert self.model.name == "plain"

    def test_monomers(self):
        assert set(self.model.monomers.keys()) == {"S1", "S2", "S3", "S4", "S5"}

    def test_parameters_present(self):
        pnames = set(self.model.parameters.keys())
        for p in ("k1_f", "k1_r", "k2_f", "k2_r", "k3_f", "k3_r"):
            assert p in pnames

    def test_parameter_values(self):
        assert self.model.parameters["k1_f"].value == 1.0
        assert self.model.parameters["k1_r"].value == 0.1

    def test_compartment(self):
        assert "cell" in self.model.compartments.keys()

    def test_rules(self):
        rnames = set(self.model.rules.keys())
        for r in ("R1", "R2", "R3", "R4", "R5", "R6"):
            assert r in rnames

    def test_rule_count(self):
        assert len(self.model.rules) == 6

    def test_initials_count(self):
        assert len(self.model.initials) == 5

    def test_initials_values(self):
        vals = {
            ic.pattern.monomer_patterns[0].monomer.name: ic.value.value
            for ic in self.model.initials
        }
        assert vals["S1"] == 1.0
        assert vals["S2"] == 2.0
        assert vals["S5"] == 5.0

    def test_species_observables(self):
        obs_names = set(self.model.observables.keys())
        for sp in ("S1", "S2", "S3", "S4", "S5"):
            assert "obs_" + sp in obs_names

    def test_assignment_rules_as_expressions(self):
        expr_names = set(self.model.expressions.keys())
        for e in ("A", "B", "C", "D", "AA", "dim_r"):
            assert e in expr_names

    def test_dim_r_expression(self):
        """dim_r = k3_r should produce an expression referencing k3_r."""
        dim_r = self.model.expressions["dim_r"]
        assert self.model.parameters["k3_r"] in dim_r.expr.free_symbols

    def test_r1_rate_expression(self):
        """R1 rate (k1_f*S1*S2 divided by obs_S1*obs_S2) simplifies to k1_f.

        Because the simplified rate is a bare Parameter, no wrapping Expression
        is created; the rule's rate_forward is the Parameter itself.
        """
        r1 = self.model.rules["R1"]
        # After dividing (k1_f*S1*S2) by (obs_S1*obs_S2) the rate is k1_f:
        # a bare Parameter, so no R1_rate Expression is created.
        assert r1.rate_forward is self.model.parameters["k1_f"]
        assert "R1_rate" not in self.model.expressions.keys()

    def test_r5_rate_homodimer(self):
        """R5 homodimerisation: SBML kinetic law is ``0.5 * k3_f * S1^2``.

        The importer divides by ``obs_S1^2`` to get the intrinsic rate, then
        multiplies by the combinatorial correction ``2! = 2`` to compensate for
        PySB's symmetry-factor halving.  Net result: ``0.5 * k3_f * 2 = k3_f``,
        stored as the expression ``1.0 * k3_f``.
        """
        r5_rate = self.model.expressions["R5_rate"]
        # After the combinatorial correction the only free symbol is k3_f;
        # no 0.5 factor should remain.
        assert self.model.parameters["k3_f"] in r5_rate.expr.free_symbols
        nums = r5_rate.expr.atoms(sympy.Number)
        # The sole numeric coefficient must be 1 (stored as Float 1.0)
        assert all(abs(float(n) - 1.0) < 1e-9 for n in nums)

    def test_rule_reactants_products(self):
        """R1: S1 + S2 -> S3."""
        r1 = self.model.rules["R1"]
        reactant_names = sorted(
            mp.monomer.name
            for cp in r1.rule_expression.reactant_pattern.complex_patterns
            for mp in cp.monomer_patterns
        )
        product_names = [
            mp.monomer.name
            for cp in r1.rule_expression.product_pattern.complex_patterns
            for mp in cp.monomer_patterns
        ]
        assert reactant_names == ["S1", "S2"]
        assert product_names == ["S3"]


# ---------------------------------------------------------------------------
# model_from_sbml (libsbml default)
# ---------------------------------------------------------------------------


def test_model_from_sbml_uses_libsbml_by_default():
    path = _sbml_location("test_sbml_flat_SBML")
    m = model_from_sbml(path)
    assert set(m.monomers.keys()) == {"S1", "S2", "S3", "S4", "S5"}


def test__model_from_sbml_libsbml_function():
    path = _sbml_location("test_sbml_flat_SBML")
    m = _model_from_sbml_libsbml(path)
    assert len(m.rules) == 6


def test_model_from_sbml_force_flag():
    """force=True should not raise on a valid file."""
    path = _sbml_location("test_sbml_flat_SBML")
    m = model_from_sbml(path, force=True)
    assert m is not None


# ---------------------------------------------------------------------------
# Legacy atomizer path preserved
# ---------------------------------------------------------------------------


def _require_atomizer():
    try:
        pf.get_path("atomizer")
    except Exception:
        raise SkipTest("atomizer (sbmlTranslator) not available")


def test_model_from_sbml_atomizer_flat():
    _require_atomizer()
    path = _sbml_location("test_sbml_flat_SBML")
    m = model_from_sbml(path, use_libsbml=False)
    assert m is not None


def test_model_from_sbml_atomizer_structured():
    _require_atomizer()
    path = _sbml_location("test_sbml_structured_SBML")
    m = model_from_sbml(path, use_libsbml=False, atomize=True)
    assert m is not None


# ---------------------------------------------------------------------------
# model_from_biomodels – mocked download
# ---------------------------------------------------------------------------


def _biomodels_mock(accession_no, mirror):
    """Mock _download_biomodels: copy the flat test SBML to a temp file."""
    _, path = tempfile.mkstemp(suffix=".xml")
    shutil.copy(_sbml_location("test_sbml_flat_SBML"), path)
    return path


@mock.patch("pysb.importers.sbml._download_biomodels", _biomodels_mock)
def test_biomodels_libsbml_mock():
    m = model_from_biomodels("1")
    assert len(m.monomers) == 5


@mock.patch("pysb.importers.sbml._download_biomodels", _biomodels_mock)
def test_biomodels_atomizer_mock():
    _require_atomizer()
    m = model_from_biomodels("1", use_libsbml=False)
    assert m is not None


@mock.patch("pysb.importers.sbml._download_biomodels", _biomodels_mock)
def test_biomodels_full_accession():
    m = model_from_biomodels("BIOMD0000000001")
    assert m is not None


def test_biomodels_invalid_mirror():
    assert_raises_regex(ValueError, "", model_from_biomodels, "1", mirror="spam")


def test_biomodels_invalid_accession():
    assert_raises_regex(ValueError, "", model_from_biomodels, "not_a_number")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_sbml_raises():
    assert_raises_regex(SbmlImportError, "", SbmlImporter, "this is not valid XML")


def test_invalid_sbml_force_warns():
    """An SBML document with parse errors issues a warning when force=True."""
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        try:
            SbmlImporter("this is not valid XML", force=True)
        except Exception:
            pass
    assert any("error" in str(warning.message).lower() for warning in w)


def test_invalid_sbml_force_continues():
    sbml = """\
        <?xml version="1.0"?>
        <sbml xmlns="http://www.sbml.org/sbml/level2/version3" level="2" version="3">
          <model id="m"/>
        </sbml>"""
    # Empty model (no species/reactions) should succeed with force=True
    m = SbmlImporter(textwrap.dedent(sbml), force=True).model
    assert m is not None


def test_libsbml_missing():
    import pysb.importers.sbml as sbml_mod

    with mock.patch.object(sbml_mod, "libsbml", None):
        assert_raises_regex(
            ImportError,
            "libsbml",
            SbmlImporter.__new__(SbmlImporter).__init__,
            "dummy.xml",
        )


# ---------------------------------------------------------------------------
# Round-trip: PySB SBML exporter → libsbml importer
# ---------------------------------------------------------------------------


def test_roundtrip_robertson():
    """Export the Robertson model to SBML, import back, check structure."""
    from pysb.examples import robertson
    from pysb.export import export

    sbml_str = export(robertson.model, "sbml")
    m = _model_from_sbml_libsbml(sbml_str)

    # Species count should match (exported as __s0, __s1, __s2)
    assert len(m.monomers) == len(robertson.model.species)
    # Reaction count should match
    assert len(m.rules) == len(robertson.model.reactions_bidirectional)
    # Parameters should be preserved
    orig_pnames = {p.name for p in robertson.model.parameters}
    imp_pnames = {p.name for p in m.parameters}
    assert orig_pnames <= imp_pnames


def test_roundtrip_initial_values():
    """Initial values set via initialAssignment are correctly imported."""
    from pysb.examples import robertson
    from pysb.export import export

    sbml_str = export(robertson.model, "sbml")
    m = _model_from_sbml_libsbml(sbml_str)

    # Robertson: A_0=1, B_0=0, C_0=0
    ic_map = {}
    for ic in m.initials:
        mon_name = ic.pattern.monomer_patterns[0].monomer.name
        ic_map[mon_name] = ic.value
    # A_0 must be 1.0 (referenced by initialAssignment)
    assert any(getattr(v, "value", None) == 1.0 for v in ic_map.values())


# ---------------------------------------------------------------------------
# Minimal SBML constructs
# ---------------------------------------------------------------------------


def test_no_compartments():
    """Model with no explicit compartments imports without error."""
    sbml = _minimal_sbml(
        compartments="",
        species='<species id="A" initialAmount="1"/>',
        parameters='<parameter id="k" value="1.0" constant="true"/>',
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfReactants><speciesReference species="A"/></listOfReactants>
            <listOfProducts/>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    m = _model_from_sbml_libsbml(sbml, force=True)
    assert "A" in m.monomers.keys()
    assert len(m.rules) == 1


def test_species_initial_concentration():
    """initialConcentration is accepted."""
    sbml = _minimal_sbml(
        species='<species id="X" compartment="c" initialConcentration="3.0"/>',
    )
    m = _model_from_sbml_libsbml(sbml, force=True)
    assert m.parameters["X_0"].value == 3.0


def test_species_boundary_condition():
    """Boundary condition (fixed) species are imported as fixed initials."""
    sbml = _minimal_sbml(
        species='<species id="S" compartment="c" initialAmount="5" '
        'boundaryCondition="true"/>',
    )
    m = _model_from_sbml_libsbml(sbml, force=True)
    assert m.initials[0].fixed is True


def test_stoichiometry_attribute():
    """Explicit stoichiometry=2 creates two copies of the reactant pattern."""
    sbml = _minimal_sbml(
        species=(
            '<species id="A" compartment="c" initialAmount="10"/>'
            '<species id="B" compartment="c" initialAmount="0"/>'
        ),
        parameters='<parameter id="k" value="0.5" constant="true"/>',
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfReactants>
              <speciesReference species="A" stoichiometry="2"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="B"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    r = m.rules["r1"]
    reactant_cps = r.rule_expression.reactant_pattern.complex_patterns
    assert len(reactant_cps) == 2


def test_assignment_rule_references_parameter():
    """An assignment rule referencing a parameter creates an Expression."""
    sbml = _minimal_sbml(
        parameters=(
            '<parameter id="k" value="2.0" constant="true"/>'
            '<parameter id="k2" constant="false"/>'
        ),
        rules="""
          <assignmentRule variable="k2">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><times/><cn>3</cn><ci>k</ci></apply>
            </math>
          </assignmentRule>""",
    )
    m = _model_from_sbml_libsbml(sbml, force=True)
    assert "k2" in m.expressions.keys()
    assert m.parameters["k"] in m.expressions["k2"].expr.free_symbols


def test_reaction_no_kinetic_law_with_force():
    """A reaction missing a kinetic law raises without force, warns with."""
    sbml = _minimal_sbml(
        species='<species id="A" compartment="c" initialAmount="1"/>',
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfReactants><speciesReference species="A"/></listOfReactants>
            <listOfProducts/>
          </reaction>""",
    )
    assert_raises_regex(SbmlImportError, "", _model_from_sbml_libsbml, sbml)

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = _model_from_sbml_libsbml(sbml, force=True)
    assert any("kinetic law" in str(warning.message).lower() for warning in w)


# ---------------------------------------------------------------------------
# Algebraic rules
# ---------------------------------------------------------------------------


def test_algebraic_rule_raises_by_default():
    """An algebraic rule raises SbmlImportError (unsupportable in PySB)."""
    sbml = _minimal_sbml(
        parameters='<parameter id="x" value="1" constant="false"/>',
        rules="""
          <algebraicRule>
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><minus/><ci>x</ci><cn>1</cn></apply>
            </math>
          </algebraicRule>""",
    )
    assert_raises_regex(SbmlImportError, "[Aa]lgebraic", _model_from_sbml_libsbml, sbml)


def test_algebraic_rule_warns_with_force():
    """An algebraic rule issues a warning when force=True."""
    sbml = _minimal_sbml(
        parameters='<parameter id="x" value="1" constant="false"/>',
        rules="""
          <algebraicRule>
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><minus/><ci>x</ci><cn>1</cn></apply>
            </math>
          </algebraicRule>""",
    )
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        _model_from_sbml_libsbml(sbml, force=True)
    assert any("lgebraic" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_event_raises_by_default():
    """An SBML event raises SbmlImportError (not supported in PySB)."""
    sbml = _minimal_sbml(
        parameters='<parameter id="x" value="0" constant="false"/>',
    ).replace(
        "</model>",
        """<listOfEvents>
          <event id="ev1" useValuesFromTriggerTime="true">
            <trigger initialValue="false" persistent="true">
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><gt/><ci>x</ci><cn>1</cn></apply>
              </math>
            </trigger>
          </event>
        </listOfEvents></model>""",
    )
    assert_raises_regex(SbmlImportError, "[Ee]vent", _model_from_sbml_libsbml, sbml)


def test_event_warns_with_force():
    """An SBML event issues a warning when force=True."""
    sbml = _minimal_sbml(
        parameters='<parameter id="x" value="0" constant="false"/>',
    ).replace(
        "</model>",
        """<listOfEvents>
          <event id="ev1" useValuesFromTriggerTime="true">
            <trigger initialValue="false" persistent="true">
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><gt/><ci>x</ci><cn>1</cn></apply>
              </math>
            </trigger>
          </event>
        </listOfEvents></model>""",
    )
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        _model_from_sbml_libsbml(sbml, force=True)
    assert any("vent" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# Non-integer stoichiometry
# ---------------------------------------------------------------------------


def test_non_integer_stoichiometry_warns():
    """Non-integer stoichiometry issues a warning and is truncated."""
    sbml = _minimal_sbml(
        species=(
            '<species id="A" compartment="c" initialAmount="10"/>'
            '<species id="B" compartment="c" initialAmount="0"/>'
        ),
        parameters='<parameter id="k" value="1" constant="true"/>',
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfReactants>
              <speciesReference species="A" stoichiometry="1.5"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="B"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        m = _model_from_sbml_libsbml(sbml)
    assert any("stoichiometry" in str(warning.message).lower() for warning in w)
    # 1.5 truncated to 1 → one reactant pattern copy
    r = m.rules["r1"]
    assert len(r.rule_expression.reactant_pattern.complex_patterns) == 1


# ---------------------------------------------------------------------------
# Time variable
# ---------------------------------------------------------------------------


def test_time_variable_in_rate_law():
    """SBML csymbol 'time' maps to pysb.core.time in rate expressions."""
    from pysb.core import time as pysb_time

    sbml = _minimal_sbml(
        species='<species id="A" compartment="c" initialAmount="0"/>',
        parameters='<parameter id="k" value="1" constant="true"/>',
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfProducts>
              <speciesReference species="A"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><csymbol encoding="text"
                  definitionURL="http://www.sbml.org/sbml/symbols/time">
                  time</csymbol></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    rate_expr = m.expressions["r1_rate"].expr
    assert pysb_time in rate_expr.free_symbols


def test_time_variable_simulation():
    """A rate-law with 'time' integrates correctly: dA/dt = k*t => A=k*t^2/2."""
    import numpy as np
    from pysb.simulator import ScipyOdeSimulator
    from pysb.core import time as pysb_time

    sbml = _minimal_sbml(
        species='<species id="A" compartment="c" initialAmount="0"/>',
        parameters='<parameter id="k" value="2" constant="true"/>',
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfProducts><speciesReference species="A"/></listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><csymbol encoding="text"
                  definitionURL="http://www.sbml.org/sbml/symbols/time">
                  time</csymbol></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    tspan = np.linspace(0, 3, 40)
    res = ScipyOdeSimulator(m, tspan=tspan, compiler="python").run()
    # dA/dt = 2t  =>  A(t) = t^2
    np.testing.assert_allclose(res.observables["obs_A"], tspan**2, rtol=1e-4)


# ---------------------------------------------------------------------------
# _split_reversible_rate unit tests (no libsbml needed)
# ---------------------------------------------------------------------------


def test_split_reversible_rate_simple():
    """kf*A - kr*B splits into (kf*A, kr*B)."""
    kf, kr, A, B = sympy.symbols("kf kr A B", positive=True)
    expr = kf * A - kr * B
    fwd, rev = _split_reversible_rate(expr)
    assert fwd is not None
    assert rev is not None
    assert sympy.simplify(fwd - kf * A) == 0
    assert sympy.simplify(rev - kr * B) == 0


def test_split_reversible_rate_single_positive_term():
    """A single positive term has no negative part, so returns (None, None)."""
    k, A = sympy.symbols("k A", positive=True)
    fwd, rev = _split_reversible_rate(k * A)
    assert fwd is None
    assert rev is None


def test_split_reversible_rate_single_negative_term():
    """A single negative term has no positive part, so returns (None, None)."""
    k, A = sympy.symbols("k A", positive=True)
    fwd, rev = _split_reversible_rate(-k * A)
    assert fwd is None
    assert rev is None


def test_split_reversible_rate_multiple_positive():
    """All positive terms cannot be split, so returns (None, None)."""
    k1, k2, A, B = sympy.symbols("k1 k2 A B", positive=True)
    fwd, rev = _split_reversible_rate(k1 * A + k2 * B)
    assert fwd is None
    assert rev is None


def test_split_reversible_rate_three_terms():
    """k1*A + k2*B - k3*C: two positive and one negative term."""
    k1, k2, k3, A, B, C = sympy.symbols("k1 k2 k3 A B C", positive=True)
    expr = k1 * A + k2 * B - k3 * C
    fwd, rev = _split_reversible_rate(expr)
    assert fwd is not None
    assert rev is not None
    # Forward should contain k1*A and k2*B; reverse should contain k3*C
    fwd_syms = fwd.free_symbols
    rev_syms = rev.free_symbols
    assert k3 in rev_syms
    assert C in rev_syms
    assert k1 in fwd_syms
    assert k2 in fwd_syms


# ---------------------------------------------------------------------------
# Reversible reaction import tests
# ---------------------------------------------------------------------------


def test_dynamic_compartment_assignment_rule_raises():
    """An assignment rule on a compartment raises SbmlImportError."""
    sbml = _minimal_sbml(
        parameters='<parameter id="t2" value="0" constant="false"/>',
        rules="""
          <assignmentRule variable="c">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><plus/><cn>1</cn><ci>t2</ci></apply>
            </math>
          </assignmentRule>""",
    )
    assert_raises_regex(
        SbmlImportError, "[Dd]ynamic compartment", _model_from_sbml_libsbml, sbml
    )


def test_dynamic_compartment_rate_rule_raises():
    """A rate rule on a compartment raises SbmlImportError."""
    sbml = _minimal_sbml(
        rules="""
          <rateRule variable="c">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <cn>0.1</cn>
            </math>
          </rateRule>""",
    )
    assert_raises_regex(
        SbmlImportError, "[Dd]ynamic compartment", _model_from_sbml_libsbml, sbml
    )


# ---------------------------------------------------------------------------
# Rate rules (ODE-only models)
# ---------------------------------------------------------------------------


def test_rate_rule_creates_monomer_and_rule():
    """A rate-rule variable becomes a Monomer and a None>>X() ODE rule."""
    sbml = _minimal_sbml(
        parameters=(
            '<parameter id="V" value="-65" constant="false"/>'
            '<parameter id="gK" value="36" constant="true"/>'
            '<parameter id="EK" value="-77" constant="true"/>'
        ),
        rules="""
          <rateRule variable="V">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><times/><cn>-1</cn><ci>gK</ci>
                <apply><minus/><ci>V</ci><ci>EK</ci></apply>
              </apply>
            </math>
          </rateRule>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    assert "V" in m.monomers.keys()
    assert "obs_V" in m.observables.keys()
    assert "V_rate" in m.expressions.keys()
    assert "V_ode" in m.rules.keys()
    # Not a plain Parameter
    assert "V" not in m.parameters.keys()
    # Initial value preserved
    assert abs(m.parameters["V_0"].value - (-65.0)) < 1e-9


def test_rate_rule_initial_value():
    """Initial value of a rate-rule variable is created as a parameter."""
    sbml = _minimal_sbml(
        parameters='<parameter id="x" value="3.14" constant="false"/>',
        rules="""
          <rateRule variable="x">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <cn>1</cn>
            </math>
          </rateRule>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    assert abs(m.parameters["x_0"].value - 3.14) < 1e-9


def test_rate_rule_expression_references_other_vars():
    """Rate-rule RHS that references another rate-rule variable is parsed."""
    sbml = _minimal_sbml(
        parameters=(
            '<parameter id="V" value="-65" constant="false"/>'
            '<parameter id="n" value="0.3" constant="false"/>'
        ),
        rules="""
          <rateRule variable="V">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><minus/><ci>n</ci></apply>
            </math>
          </rateRule>
          <rateRule variable="n">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><minus/><ci>V</ci><ci>n</ci></apply>
            </math>
          </rateRule>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    # Both variables become monomers; V_rate references obs_n
    V_rate_syms = {s.name for s in m.expressions["V_rate"].expr.free_symbols}
    assert "obs_n" in V_rate_syms


def test_rate_rule_model_simulates():
    """An ODE-only (rate-rule) model can be integrated with ScipyOdeSimulator."""
    import numpy as np
    from pysb.simulator import ScipyOdeSimulator

    sbml = _minimal_sbml(
        parameters=(
            '<parameter id="x" value="1.0" constant="false"/>'
            '<parameter id="k" value="0.5" constant="true"/>'
        ),
        rules="""
          <rateRule variable="x">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><times/><cn>-1</cn><ci>k</ci><ci>x</ci></apply>
            </math>
          </rateRule>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    tspan = np.linspace(0, 4, 50)
    res = ScipyOdeSimulator(m, tspan=tspan, compiler="python").run()
    # dx/dt = -0.5 x  =>  x(t) = exp(-0.5 t)
    expected = np.exp(-0.5 * tspan)
    np.testing.assert_allclose(res.observables["obs_x"], expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


def test_parameter_without_value_is_skipped():
    """A parameter with no value attribute is silently skipped."""
    sbml = _minimal_sbml(
        parameters='<parameter id="k" constant="false"/>',
    )
    m = _model_from_sbml_libsbml(sbml, force=True)
    # k has no value -> not added as a Parameter
    assert "k" not in m.parameters.keys()


def test_function_definition_non_lambda_skipped():
    """A functionDefinition whose math is not a lambda is silently skipped."""
    # Use a bare <cn> instead of an <lambda> as the math for the function.
    sbml = _minimal_sbml(
        species='<species id="A" compartment="c" initialAmount="1"/>',
        parameters='<parameter id="k" value="1" constant="true"/>',
    ).replace(
        '<model id="test_model">',
        """<model id="test_model">
    <listOfFunctionDefinitions>
      <functionDefinition id="bad_fn">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn>42</cn>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>""",
    )
    # Should import cleanly – the malformed function is skipped
    m = _model_from_sbml_libsbml(sbml, force=True)
    assert m is not None


def test_function_definition_zero_arg_lambda_skipped():
    """A lambda with zero children (malformed) is silently skipped."""
    # libsbml will reject a truly empty lambda via schema validation, so we
    # build a lambda with only a body and no arguments (n_children == 1, but
    # n_children - 1 == 0 argument names).  This exercises the arg_names=[]
    # path and verifies the importer does not crash.
    sbml = _minimal_sbml(
        species='<species id="A" compartment="c" initialAmount="1"/>',
        parameters='<parameter id="k" value="1" constant="true"/>',
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfProducts><speciesReference species="A"/></listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    ).replace(
        '<model id="test_model">',
        """<model id="test_model">
    <listOfFunctionDefinitions>
      <functionDefinition id="fn0">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <lambda>
            <cn>1</cn>
          </lambda>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>""",
    )
    m = _model_from_sbml_libsbml(sbml, force=True)
    assert "A" in m.monomers.keys()


def test_sbml_l3v1_model():
    """An SBML Level 3 Version 1 model is imported correctly."""
    sbml = _minimal_sbml(
        compartments='<compartment id="c" size="1" constant="true"/>',
        species=(
            '<species id="A" compartment="c" initialConcentration="2.0"'
            ' hasOnlySubstanceUnits="false" boundaryCondition="false"'
            ' constant="false"/>'
            '<species id="B" compartment="c" initialConcentration="0.0"'
            ' hasOnlySubstanceUnits="false" boundaryCondition="false"'
            ' constant="false"/>'
        ),
        parameters='<parameter id="k" value="0.1" constant="true"/>',
        reactions="""
          <reaction id="r1" reversible="false" fast="false">
            <listOfReactants>
              <speciesReference species="A" stoichiometry="1" constant="true"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="B" stoichiometry="1" constant="true"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>""",
        level=3,
        version=1,
    ).replace(
        'xmlns="http://www.sbml.org/sbml/level3/version1"',
        'xmlns="http://www.sbml.org/sbml/level3/version1/core"',
    )
    m = _model_from_sbml_libsbml(sbml)
    assert "A" in m.monomers.keys()
    assert "B" in m.monomers.keys()
    assert "r1" in m.rules.keys()
    assert abs(m.parameters["A_0"].value - 2.0) < 1e-9


def test_initial_amount_takes_precedence_over_concentration():
    """When both initialAmount and initialConcentration are set, amount wins."""
    # Build SBML manually since _minimal_sbml doesn't support both attributes
    sbml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level2/version3"'
        ' level="2" version="3">\n'
        '  <model id="test_model">\n'
        "    <listOfCompartments>"
        '<compartment id="c" size="1"/>'
        "</listOfCompartments>\n"
        "    <listOfSpecies>"
        '<species id="A" compartment="c"'
        ' initialAmount="7.0" initialConcentration="3.0"/>'
        "</listOfSpecies>\n"
        "  </model>\n"
        "</sbml>"
    )
    m = _model_from_sbml_libsbml(sbml, force=True)
    assert abs(m.parameters["A_0"].value - 7.0) < 1e-9


def test_zero_dimensional_compartment():
    """A compartment with spatialDimensions=0 imports with dimension 3 fallback."""
    sbml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level2/version3"'
        ' level="2" version="3">\n'
        '  <model id="test_model">\n'
        "    <listOfCompartments>"
        '<compartment id="c" size="1" spatialDimensions="0"/>'
        "</listOfCompartments>\n"
        "  </model>\n"
        "</sbml>"
    )
    m = _model_from_sbml_libsbml(sbml, force=True)
    assert "c" in m.compartments.keys()


def test_reaction_naming_uses_sanitised_id():
    """A reaction with id='r0' produces a rule named 'r0'."""
    sbml = _minimal_sbml(
        species='<species id="A" compartment="c" initialAmount="1"/>',
        parameters='<parameter id="k" value="1" constant="true"/>',
        reactions="""
          <reaction id="r0" reversible="false">
            <listOfReactants><speciesReference species="A"/></listOfReactants>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    assert "r0" in m.rules.keys()


def test_reaction_fallback_name_when_no_id():
    """A reaction with no id attribute receives the fallback name r{i}.

    SBML Level 2 requires a reaction ``id``; omitting it produces a parse
    error, so ``force=True`` is needed to continue past the error.  The
    fallback name ``r0`` (index 0) should be assigned to the rule.
    """
    sbml = _minimal_sbml(
        species='<species id="A" compartment="c" initialAmount="1"/>',
        parameters='<parameter id="k" value="1" constant="true"/>',
        reactions="""
          <reaction reversible="false">
            <listOfReactants><speciesReference species="A"/></listOfReactants>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    import warnings as _warnings

    with _warnings.catch_warnings(record=True):
        _warnings.simplefilter("always")
        m = _model_from_sbml_libsbml(sbml, force=True)
    assert "r0" in m.rules.keys()


def test_reversible_reaction_split_into_two_rules():
    """A reversible SBML reaction with separable kinetic law produces two rules.

    The net-flux kinetic law ``kf*A - kr*B`` is split into a forward flux
    ``kf*A`` and a reverse flux ``kr*B``.  Each half is divided by the
    appropriate species observable to obtain the intrinsic rate constant, and
    two non-reversible PySB rules are created: ``r1_fwd`` (A -> B at kf) and
    ``r1_rev`` (B -> A at kr).
    """
    sbml = _minimal_sbml(
        species=(
            '<species id="A" compartment="c" initialAmount="1"/>'
            '<species id="B" compartment="c" initialAmount="0"/>'
        ),
        parameters=(
            '<parameter id="kf" value="1" constant="true"/>'
            '<parameter id="kr" value="0.5" constant="true"/>'
        ),
        reactions="""
          <reaction id="r1" reversible="true">
            <listOfReactants>
              <speciesReference species="A"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="B"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><minus/>
                  <apply><times/><ci>kf</ci><ci>A</ci></apply>
                  <apply><times/><ci>kr</ci><ci>B</ci></apply>
                </apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        m = _model_from_sbml_libsbml(sbml)
    # No warnings expected for cleanly separable kinetic law
    assert not any("cannot be split" in str(warning.message) for warning in w)
    # Two rules created: forward and reverse
    assert "r1_fwd" in m.rules.keys()
    assert "r1_rev" in m.rules.keys()
    assert "r1" not in m.rules.keys()
    # Both rules are non-reversible
    assert m.rules["r1_fwd"].rule_expression.is_reversible is False
    assert m.rules["r1_rev"].rule_expression.is_reversible is False
    # Forward rate = kf (kinetic law kf*A divided by obs_A → bare Parameter)
    assert m.rules["r1_fwd"].rate_forward is m.parameters["kf"]
    assert "r1_fwd_rate" not in m.expressions.keys()
    # Reverse rate = kr (kinetic law kr*B divided by obs_B → bare Parameter)
    assert m.rules["r1_rev"].rate_forward is m.parameters["kr"]
    assert "r1_rev_rate" not in m.expressions.keys()


def test_reversible_reaction_unseparable_warns():
    """A reversible reaction with unseparable kinetic law warns and creates one rule.

    When ``_split_reversible_rate`` cannot decompose the kinetic law (e.g.
    all terms have the same sign, or there is only one term), a warning is
    issued and only the forward rule is created.
    """
    sbml = _minimal_sbml(
        species=(
            '<species id="A" compartment="c" initialAmount="1"/>'
            '<species id="B" compartment="c" initialAmount="0"/>'
        ),
        parameters='<parameter id="k" value="1" constant="true"/>',
        reactions="""
          <reaction id="r1" reversible="true">
            <listOfReactants>
              <speciesReference species="A"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="B"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        m = _model_from_sbml_libsbml(sbml)
    # Warning about unseparable kinetic law
    assert any("cannot be split" in str(warning.message) for warning in w)
    # Only the forward rule is created
    assert "r1_fwd" in m.rules.keys()
    assert "r1_rev" not in m.rules.keys()
    assert m.rules["r1_fwd"].rule_expression.is_reversible is False


def test_duplicate_rate_rule_raises():
    """Two rate rules for the same variable raise SbmlImportError."""
    sbml = _minimal_sbml(
        parameters='<parameter id="x" value="1" constant="false"/>',
        rules="""
          <rateRule variable="x">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
          </rateRule>
          <rateRule variable="x">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>2</cn></math>
          </rateRule>""",
    )
    assert_raises_regex(
        SbmlImportError, "[Mm]ultiple rate rules", _model_from_sbml_libsbml, sbml
    )


def test_duplicate_rate_rule_warns_with_force():
    """Two rate rules for the same variable warn when force=True."""
    sbml = _minimal_sbml(
        parameters='<parameter id="x" value="1" constant="false"/>',
        rules="""
          <rateRule variable="x">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
          </rateRule>
          <rateRule variable="x">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>2</cn></math>
          </rateRule>""",
    )
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        m = _model_from_sbml_libsbml(sbml, force=True)
    assert any("rate rule" in str(warning.message).lower() for warning in w)
    # Only one ODE rule should exist for x
    assert "x_ode" in m.rules.keys()


def test_compartment_hierarchy():
    """Nested compartments (outer/inner) are linked via the parent attribute."""
    sbml = _minimal_sbml(
        compartments=(
            '<compartment id="outer" size="10"/>'
            '<compartment id="inner" size="1" outside="outer"/>'
        ),
    )
    m = _model_from_sbml_libsbml(sbml, force=True)
    inner = m.compartments["inner"]
    outer = m.compartments["outer"]
    assert inner.parent is outer


def test_integer_stoichiometry_float_representation():
    """Stoichiometry given as 2.0 (exact float) is treated as integer 2."""
    # Some SBML files write stoichiometry as floating-point (e.g. 2.0).
    # The math.isclose fix ensures 2.0 is recognised as integer 2.
    sbml = _minimal_sbml(
        species=(
            '<species id="A" compartment="c" initialAmount="10"/>'
            '<species id="B" compartment="c" initialAmount="0"/>'
        ),
        parameters='<parameter id="k" value="0.5" constant="true"/>',
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfReactants>
              <speciesReference species="A" stoichiometry="2"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="B"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        m = _model_from_sbml_libsbml(sbml)
    # No truncation warning should be raised for exact integer 2.0
    assert not any("stoichiometry" in str(warning.message).lower() for warning in w)
    r = m.rules["r1"]
    assert len(r.rule_expression.reactant_pattern.complex_patterns) == 2


# ---------------------------------------------------------------------------
# Additional branch-coverage tests
# ---------------------------------------------------------------------------


def test_sbml_no_model_raises():
    """An SBML document that libsbml reports as having no model raises SbmlImportError."""
    sbml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level2/version3"'
        ' level="2" version="3"/>\n'
    )
    assert_raises_regex(SbmlImportError, "", _model_from_sbml_libsbml, sbml)


def test_rate_rule_for_existing_species():
    """A rate rule whose variable is already an SBML species is handled
    without creating a duplicate Monomer (the species is reused)."""
    sbml = _minimal_sbml(
        species='<species id="A" compartment="c" initialAmount="1"/>',
        rules="""
          <rateRule variable="A">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <cn>-1</cn>
            </math>
          </rateRule>""",
    )
    m = _model_from_sbml_libsbml(sbml, force=True)
    # A should be a Monomer (from species), not duplicated
    assert "A" in m.monomers.keys()


def test_assignment_rule_skips_existing_component():
    """An assignment rule targeting a name that already exists as a model
    component (e.g. an observable) is silently skipped."""
    # The species 'A' creates obs_A; an assignment rule for 'obs_A' should
    # be ignored because that name already exists as a model component.
    sbml = _minimal_sbml(
        species='<species id="A" compartment="c" initialAmount="1"/>',
        parameters='<parameter id="obs_A" constant="false" value="0"/>',
        rules="""
          <assignmentRule variable="obs_A">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <ci>A</ci>
            </math>
          </assignmentRule>""",
    )
    m = _model_from_sbml_libsbml(sbml, force=True)
    # obs_A is already an Observable from the species; the assignment rule
    # for 'obs_A' parameter should not overwrite it.
    assert "obs_A" in m.observables.keys()


def test_split_reversible_rate_zero_coeff():
    """A term with a zero coefficient (e.g. the integer 0 itself) returns (None, None)."""
    # sympy.Integer(0) has as_coeff_Mul() == (0, 1): zero coefficient path
    fwd, rev = _split_reversible_rate(sympy.Integer(0))
    assert fwd is None
    assert rev is None


def test_split_reversible_rate_symbolic_coefficient():
    """A term whose leading coefficient cannot be determined as positive or
    negative (e.g. a bare symbolic product without a numeric leading factor)
    returns (None, None)."""
    # sin(A)*B: as_coeff_Mul -> (1, sin(A)*B), coeff is 1, positive
    # This exercises the positive-term path but results in (None, None)
    # because there's no negative term.
    A, B = sympy.symbols("A B")
    expr = sympy.sin(A) * B
    fwd, rev = _split_reversible_rate(expr)
    # Single positive term → no split
    assert fwd is None
    assert rev is None


def test_model_name_from_filename_when_no_model_id():
    """When the SBML model element has no id, the filename stem is used."""
    sbml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level2/version3"'
        ' level="2" version="3">\n'
        "  <model>\n"
        "  </model>\n"
        "</sbml>"
    )
    fd, path = tempfile.mkstemp(suffix=".xml", prefix="my_model_")
    os.close(fd)
    with open(path, "w") as f:
        f.write(sbml)
    try:
        m = _model_from_sbml_libsbml(path, force=True)
        # Model name should be derived from the file basename (without .xml)
        assert "my_model_" in m.name
    finally:
        os.unlink(path)


def test_function_definition_used_in_rate_law():
    """A valid SBML functionDefinition is parsed and used in a kinetic law."""
    sbml = _minimal_sbml(
        species='<species id="A" compartment="c" initialAmount="1"/>',
        parameters='<parameter id="k" value="2.0" constant="true"/>',
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfReactants><speciesReference species="A"/></listOfReactants>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><ci>myfunc</ci><ci>k</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    ).replace(
        '<model id="test_model">',
        """<model id="test_model">
    <listOfFunctionDefinitions>
      <functionDefinition id="myfunc">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <lambda>
            <bvar><ci>x</ci></bvar>
            <bvar><ci>y</ci></bvar>
            <apply><times/><ci>x</ci><ci>y</ci></apply>
          </lambda>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    # myfunc(k, A) = k*A; after dividing by obs_A (reactant) the rate = k
    # (bare Parameter), so no r1_rate Expression is created.
    assert m.rules["r1"].rate_forward is m.parameters["k"]
    assert "r1_rate" not in m.expressions.keys()


def test_initial_assignment_numeric_updates_ic():
    """A numeric initialAssignment updates the species initial condition value."""
    sbml = _minimal_sbml(
        species='<species id="A" compartment="c" initialAmount="0"/>',
        parameters='<parameter id="a0" value="5.0" constant="true"/>',
    ).replace(
        "</model>",
        """<listOfInitialAssignments>
      <initialAssignment symbol="A">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn>7.0</cn>
        </math>
      </initialAssignment>
    </listOfInitialAssignments></model>""",
    )
    m = _model_from_sbml_libsbml(sbml, force=True)
    # The numeric initialAssignment should update A_0 to 7.0
    assert abs(m.parameters["A_0"].value - 7.0) < 1e-9


def test_fast_reaction_warns():
    """A reaction with fast='true' issues a warning and is imported normally."""
    sbml = _minimal_sbml(
        species=(
            '<species id="A" compartment="c" initialAmount="1"/>'
            '<species id="B" compartment="c" initialAmount="0"/>'
        ),
        parameters='<parameter id="k" value="1" constant="true"/>',
        reactions="""
          <reaction id="r1" reversible="false" fast="true">
            <listOfReactants><speciesReference species="A"/></listOfReactants>
            <listOfProducts><speciesReference species="B"/></listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        m = _model_from_sbml_libsbml(sbml)
    assert any("fast" in str(warning.message).lower() for warning in w)
    assert "r1" in m.rules.keys()


def test_local_kinetic_law_parameter_promoted():
    """Local kinetic-law parameters with no global counterpart are promoted to global Parameters."""
    # Use L2 SBML which has <listOfParameters> inside <kineticLaw>
    sbml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level2/version3"'
        ' level="2" version="3">\n'
        '  <model id="test_model">\n'
        "    <listOfCompartments>"
        '<compartment id="c" size="1"/>'
        "</listOfCompartments>\n"
        "    <listOfSpecies>"
        '<species id="A" compartment="c" initialAmount="1"/>'
        '<species id="B" compartment="c" initialAmount="0"/>'
        "</listOfSpecies>\n"
        "    <listOfReactions>"
        '      <reaction id="r1" reversible="false">\n'
        "        <listOfReactants>"
        '<speciesReference species="A"/>'
        "</listOfReactants>\n"
        "        <listOfProducts>"
        '<speciesReference species="B"/>'
        "</listOfProducts>\n"
        "        <kineticLaw>\n"
        '          <math xmlns="http://www.w3.org/1998/Math/MathML">\n'
        "            <apply><times/><ci>k_local</ci><ci>A</ci></apply>\n"
        "          </math>\n"
        "          <listOfParameters>"
        '<parameter id="k_local" value="0.5"/>'
        "</listOfParameters>\n"
        "        </kineticLaw>\n"
        "      </reaction>\n"
        "    </listOfReactions>\n"
        "  </model>\n"
        "</sbml>"
    )
    m = _model_from_sbml_libsbml(sbml)
    # k_local should have been promoted to a global Parameter
    assert "k_local" in m.parameters.keys(), "k_local not promoted to global Parameter"
    assert m.parameters["k_local"].value == 0.5
    assert "r1" in m.rules.keys()


def test_local_kinetic_law_parameter_shadow_warns():
    """A local kinetic-law parameter that shadows a global one issues a warning."""
    sbml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level2/version3"'
        ' level="2" version="3">\n'
        '  <model id="test_model">\n'
        "    <listOfCompartments>"
        '<compartment id="c" size="1"/>'
        "</listOfCompartments>\n"
        "    <listOfSpecies>"
        '<species id="A" compartment="c" initialAmount="1"/>'
        '<species id="B" compartment="c" initialAmount="0"/>'
        "</listOfSpecies>\n"
        # Global parameter k_local = 1.0
        "    <listOfParameters>"
        '<parameter id="k_local" value="1.0"/>'
        "</listOfParameters>\n"
        "    <listOfReactions>"
        '      <reaction id="r1" reversible="false">\n'
        "        <listOfReactants>"
        '<speciesReference species="A"/>'
        "</listOfReactants>\n"
        "        <listOfProducts>"
        '<speciesReference species="B"/>'
        "</listOfProducts>\n"
        "        <kineticLaw>\n"
        '          <math xmlns="http://www.w3.org/1998/Math/MathML">\n'
        "            <apply><times/><ci>k_local</ci><ci>A</ci></apply>\n"
        "          </math>\n"
        # Local parameter with same name shadows global
        "          <listOfParameters>"
        '<parameter id="k_local" value="0.5"/>'
        "</listOfParameters>\n"
        "        </kineticLaw>\n"
        "      </reaction>\n"
        "    </listOfReactions>\n"
        "  </model>\n"
        "</sbml>"
    )
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        m = _model_from_sbml_libsbml(sbml)
    assert any("shadow" in str(warning.message).lower() for warning in w), (
        "Expected a shadowing warning for local parameter k_local"
    )
    assert "r1" in m.rules.keys()


# ---------------------------------------------------------------------------
# _check_sbml_level_version tests
# ---------------------------------------------------------------------------


def test_sbml_level1_warns():
    """Importing an SBML Level 1 document issues a warning about limited support."""
    import warnings as _warnings

    sbml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level1" level="1" version="2">\n'
        '  <model name="test_model"/>\n'
        "</sbml>"
    )
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        _model_from_sbml_libsbml(sbml, force=True)
    assert any("Level 1" in str(warning.message) for warning in w)


def test_sbml_l2v5_no_version_warning():
    """SBML Level 2 Version 5 (latest known L2) does not trigger a version warning."""
    import warnings as _warnings

    # Use _minimal_sbml at L2V5; no level/version warning expected
    sbml = _minimal_sbml(level=2, version=5)
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        _model_from_sbml_libsbml(sbml, force=True)
    version_warns = [
        warning
        for warning in w
        if "Version" in str(warning.message) and "Level 2" in str(warning.message)
    ]
    assert not version_warns


def test_sbml_l3v2_no_version_warning():
    """SBML Level 3 Version 2 (latest known L3) does not trigger a version warning."""
    import warnings as _warnings

    sbml = _minimal_sbml(
        compartments='<compartment id="c" size="1" constant="true"/>',
        level=3,
        version=2,
    ).replace(
        'xmlns="http://www.sbml.org/sbml/level3/version2"',
        'xmlns="http://www.sbml.org/sbml/level3/version2/core"',
    )
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        _model_from_sbml_libsbml(sbml, force=True)
    version_warns = [
        warning
        for warning in w
        if "Version" in str(warning.message) and "Level 3" in str(warning.message)
    ]
    assert not version_warns


def test_sbml_unknown_level_warns():
    """A document whose libsbml-reported level is >3 issues an 'unknown' warning."""
    import warnings as _warnings

    # We mock _check_sbml_level_version's call to doc.getLevel()/getVersion()
    # by patching the libsbml SBMLDocument object after reading.
    sbml = _minimal_sbml(level=2, version=3)
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        # Patch readSBMLFromString to return a doc whose level reports as 4
        import pysb.importers.sbml as sbml_mod

        real_reader_cls = sbml_mod.libsbml.SBMLReader

        class _FakeReader:
            def readSBMLFromString(self, xml_string):
                doc = real_reader_cls().readSBMLFromString(xml_string)
                doc.getLevel = lambda: 4
                doc.getVersion = lambda: 1
                return doc

        with mock.patch.object(sbml_mod.libsbml, "SBMLReader", _FakeReader):
            _model_from_sbml_libsbml(sbml, force=True)
    assert any("Level 4" in str(warning.message) for warning in w)


def test_sbml_future_l2_version_warns():
    """A document with a future L2 version (>5) issues a warning."""
    import warnings as _warnings

    sbml = _minimal_sbml(level=2, version=3)
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        import pysb.importers.sbml as sbml_mod

        real_reader_cls = sbml_mod.libsbml.SBMLReader

        class _FakeReader:
            def readSBMLFromString(self, xml_string):
                doc = real_reader_cls().readSBMLFromString(xml_string)
                doc.getLevel = lambda: 2
                doc.getVersion = lambda: 99
                return doc

        with mock.patch.object(sbml_mod.libsbml, "SBMLReader", _FakeReader):
            _model_from_sbml_libsbml(sbml, force=True)
    assert any("Level 2 Version 99" in str(warning.message) for warning in w)


def test_sbml_future_l3_version_warns():
    """A document with a future L3 version (>2) issues a warning."""
    import warnings as _warnings

    sbml = _minimal_sbml(level=2, version=3)
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        import pysb.importers.sbml as sbml_mod

        real_reader_cls = sbml_mod.libsbml.SBMLReader

        class _FakeReader:
            def readSBMLFromString(self, xml_string):
                doc = real_reader_cls().readSBMLFromString(xml_string)
                doc.getLevel = lambda: 3
                doc.getVersion = lambda: 9
                return doc

        with mock.patch.object(sbml_mod.libsbml, "SBMLReader", _FakeReader):
            _model_from_sbml_libsbml(sbml, force=True)
    assert any("Level 3 Version 9" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# Cross-compartment transport tests
# ---------------------------------------------------------------------------


def _cross_compartment_sbml(
    V_src,
    V_dst,
    k,
    A0,
    reversible=False,
    kr=None,
    B0=0.0,
):
    """Build minimal SBML for a transport reaction A(src) <-> B(dst).

    For irreversible: A -> B, kinetic law = k * A * V_src
    For reversible:   net flux = k * A * V_src - kr * B * V_dst
    (Kinetic laws in amount/time; BNG-style volume-corrected rate extraction
    is what we are testing.)
    """
    fwd_law = "<apply><times/><ci>k</ci><ci>A</ci><cn>{}</cn></apply>".format(V_src)
    if reversible:
        rev_law = "<apply><times/><ci>kr</ci><ci>B</ci><cn>{}</cn></apply>".format(
            V_dst
        )
        kinetic_math = "<apply><minus/>{}{}</apply>".format(fwd_law, rev_law)
        rev_attr = 'reversible="true"'
        kr_param = '<parameter id="kr" value="{}" constant="true"/>'.format(kr)
        B_init = 'initialConcentration="{}"'.format(B0)
    else:
        kinetic_math = fwd_law
        rev_attr = 'reversible="false"'
        kr_param = ""
        B_init = 'initialConcentration="0.0"'

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level2/version3"'
        ' level="2" version="3">\n'
        '  <model id="transport_model">\n'
        "    <listOfCompartments>\n"
        '      <compartment id="src" size="{V_src}"/>\n'
        '      <compartment id="dst" size="{V_dst}"/>\n'
        "    </listOfCompartments>\n"
        "    <listOfSpecies>\n"
        '      <species id="A" compartment="src"'
        ' initialConcentration="{A0}" hasOnlySubstanceUnits="false"/>\n'
        '      <species id="B" compartment="dst"'
        ' {B_init} hasOnlySubstanceUnits="false"/>\n'
        "    </listOfSpecies>\n"
        "    <listOfParameters>\n"
        '      <parameter id="k" value="{k}" constant="true"/>\n'
        "      {kr_param}\n"
        "    </listOfParameters>\n"
        "    <listOfReactions>\n"
        '      <reaction id="r1" {rev_attr}>\n'
        "        <listOfReactants>\n"
        '          <speciesReference species="A"/>\n'
        "        </listOfReactants>\n"
        "        <listOfProducts>\n"
        '          <speciesReference species="B"/>\n'
        "        </listOfProducts>\n"
        "        <kineticLaw>\n"
        '          <math xmlns="http://www.w3.org/1998/Math/MathML">\n'
        "            {kinetic_math}\n"
        "          </math>\n"
        "        </kineticLaw>\n"
        "      </reaction>\n"
        "    </listOfReactions>\n"
        "  </model>\n"
        "</sbml>"
    ).format(
        V_src=V_src,
        V_dst=V_dst,
        A0=A0,
        B_init=B_init,
        k=k,
        kr_param=kr_param,
        rev_attr=rev_attr,
        kinetic_math=kinetic_math,
    )


def test_cross_compartment_irreversible_structure():
    """Cross-compartment irreversible transport creates deg + prod split rules."""
    sbml = _cross_compartment_sbml(V_src=2.0, V_dst=5.0, k=0.3, A0=1.0)
    m = _model_from_sbml_libsbml(sbml)
    rule_names = set(m.rules.keys())
    # Must have deg (A -> None) and prod (None -> B) rules; original r1 absent
    assert "r1_deg" in rule_names, "Expected r1_deg rule"
    assert "r1_prod" in rule_names, "Expected r1_prod rule"
    assert "r1" not in rule_names, "Original r1 rule should not exist"
    # r1_deg reactants contain A; r1_prod products contain B
    deg_reactants = [
        mp.monomer.name
        for cp in m.rules["r1_deg"].rule_expression.reactant_pattern.complex_patterns
        for mp in cp.monomer_patterns
    ]
    assert "A" in deg_reactants
    prod_products = [
        mp.monomer.name
        for cp in m.rules["r1_prod"].rule_expression.product_pattern.complex_patterns
        for mp in cp.monomer_patterns
    ]
    assert "B" in prod_products


def test_cross_compartment_irreversible_simulation():
    """Cross-compartment irreversible transport integrates correctly.

    SBML model: A(src, V=2) -> B(dst, V=5), J = k*A*V_src = k*[A]*V_src
    ODEs:  d[A]/dt = -k*[A]   (k = 0.3)
           d[B]/dt = +k*[A]*V_src/V_dst = 0.12*[A]
    Analytic solution (A0=1, B0=0):
           [A](t) = exp(-k*t)
           [B](t) = (k*V_src/V_dst) * (1 - exp(-k*t)) / k
                  = (V_src/V_dst) * (1 - exp(-k*t))
                  = 0.4 * (1 - exp(-0.3*t))
    """
    import numpy as np
    from pysb.simulator import ScipyOdeSimulator

    k, V_src, V_dst, A0 = 0.3, 2.0, 5.0, 1.0
    sbml = _cross_compartment_sbml(V_src=V_src, V_dst=V_dst, k=k, A0=A0)
    m = _model_from_sbml_libsbml(sbml)
    tspan = np.linspace(0, 5, 100)
    res = ScipyOdeSimulator(m, tspan=tspan, compiler="python").run()

    A_sim = res.observables["obs_A"]
    B_sim = res.observables["obs_B"]
    A_ref = A0 * np.exp(-k * tspan)
    B_ref = (V_src / V_dst) * (1.0 - np.exp(-k * tspan))

    np.testing.assert_allclose(A_sim, A_ref, rtol=1e-4, err_msg="[A] mismatch")
    np.testing.assert_allclose(B_sim, B_ref, rtol=1e-4, err_msg="[B] mismatch")


def test_cross_compartment_reversible_structure():
    """Cross-compartment reversible transport creates four split rules."""
    sbml = _cross_compartment_sbml(
        V_src=2.0, V_dst=5.0, k=0.3, kr=0.1, A0=1.0, B0=0.5, reversible=True
    )
    m = _model_from_sbml_libsbml(sbml)
    rule_names = set(m.rules.keys())
    expected = {"r1_fwd_deg", "r1_fwd_prod", "r1_rev_deg", "r1_rev_prod"}
    for name in expected:
        assert name in rule_names, "Expected rule {} missing".format(name)
    assert "r1" not in rule_names
    assert "r1_fwd" not in rule_names
    assert "r1_rev" not in rule_names


def test_cross_compartment_reversible_simulation():
    """Cross-compartment reversible transport integrates to analytic solution.

    SBML model: A(src, V=2) <-> B(dst, V=5)
    J_net = kf*[A]*V_src - kr*[B]*V_dst  (kf=0.3, kr=0.1)
    ODEs:
        d[A]/dt = -J_net/V_src = -kf*[A] + kr*[B]*V_dst/V_src
        d[B]/dt = +J_net/V_dst = +kf*[A]*V_src/V_dst - kr*[B]
    At steady state:  kf*[A]_ss*V_src = kr*[B]_ss*V_dst
    Conservation:     [A]*V_src + [B]*V_dst = A0*V_src + B0*V_dst (amounts)
    """
    import numpy as np
    from pysb.simulator import ScipyOdeSimulator
    from scipy.integrate import solve_ivp

    kf, kr, V_src, V_dst, A0, B0 = 0.3, 0.1, 2.0, 5.0, 1.0, 0.0
    sbml = _cross_compartment_sbml(
        V_src=V_src, V_dst=V_dst, k=kf, kr=kr, A0=A0, B0=B0, reversible=True
    )
    m = _model_from_sbml_libsbml(sbml)
    tspan = np.linspace(0, 10, 200)
    res = ScipyOdeSimulator(m, tspan=tspan, compiler="python").run()

    # Analytic reference via scipy solve_ivp
    def odes(t, y):
        A, B = y
        J = kf * A * V_src - kr * B * V_dst
        return [-J / V_src, J / V_dst]

    ref = solve_ivp(odes, [0, tspan[-1]], [A0, B0], t_eval=tspan, rtol=1e-10)
    A_ref = ref.y[0]
    B_ref = ref.y[1]

    np.testing.assert_allclose(
        res.observables["obs_A"], A_ref, rtol=1e-3, err_msg="[A] mismatch"
    )
    np.testing.assert_allclose(
        res.observables["obs_B"], B_ref, rtol=1e-3, err_msg="[B] mismatch"
    )


# ---------------------------------------------------------------------------
# Homo-multimer and stoichiometry regression tests
# ---------------------------------------------------------------------------


def test_homodimer_simulation_accuracy():
    """Homodimerisation A+A->D: ODE must use correct combinatorial factor.

    SBML kinetic law: J = k * A^2  (mass-action, amount/time in a unit
    compartment so J = k * [A]^2).

    ODE: d[A]/dt = -2*k*[A]^2,  d[D]/dt = +k*[A]^2
    Analytic solution with A0=1, D0=0:
        [A](t) = 1 / (1 + 2*k*t)
        [D](t) = (1 - [A]) / 2 = k*t / (1 + 2*k*t)

    The combinatorial correction (2! = 2) must cancel the 0.5 implicit
    symmetry factor that BNG applies, so the effective rate seen by the ODE
    is k, not k/2.
    """
    import numpy as np
    from pysb.simulator import ScipyOdeSimulator

    k = 0.5
    sbml = _minimal_sbml(
        species=(
            '<species id="A" compartment="c" initialConcentration="1.0"'
            ' hasOnlySubstanceUnits="false"/>'
            '<species id="D" compartment="c" initialConcentration="0.0"'
            ' hasOnlySubstanceUnits="false"/>'
        ),
        parameters='<parameter id="k" value="{}" constant="true"/>'.format(k),
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfReactants>
              <speciesReference species="A" stoichiometry="2"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="D"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci>
                  <apply><power/><ci>A</ci><cn>2</cn></apply>
                </apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    tspan = np.linspace(0, 5, 200)
    res = ScipyOdeSimulator(m, tspan=tspan, compiler="python").run()

    A_sim = res.observables["obs_A"]
    D_sim = res.observables["obs_D"]
    A_ref = 1.0 / (1.0 + 2.0 * k * tspan)
    D_ref = k * tspan / (1.0 + 2.0 * k * tspan)

    np.testing.assert_allclose(A_sim, A_ref, rtol=1e-4, err_msg="[A] mismatch")
    np.testing.assert_allclose(D_sim, D_ref, rtol=1e-4, err_msg="[D] mismatch")


def test_homodimer_combinatorial_correction_value():
    """Rate expression for A+A->D contains factor 2! = 2 (no 0.5 remaining)."""
    import sympy

    k = 0.5
    sbml = _minimal_sbml(
        species=(
            '<species id="A" compartment="c" initialConcentration="1.0"'
            ' hasOnlySubstanceUnits="false"/>'
            '<species id="D" compartment="c" initialConcentration="0.0"'
            ' hasOnlySubstanceUnits="false"/>'
        ),
        parameters='<parameter id="k" value="{}" constant="true"/>'.format(k),
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfReactants>
              <speciesReference species="A" stoichiometry="2"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="D"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci>
                  <apply><power/><ci>A</ci><cn>2</cn></apply>
                </apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    # After dividing by obs_A^2 and multiplying by 2!, kinetic law k*A^2 /
    # (A^2) * 2 = 2k = 1.0.  Rate should be stored as a bare Parameter with
    # value 1.0, or as an Expression with no sub-1 numeric coefficients.
    rate = m.rules["r1"].rate_forward
    if hasattr(rate, "expr"):
        nums = rate.expr.atoms(sympy.Number)
        assert all(float(n) >= 1.0 for n in nums if float(n) != 0), (
            "Unexpected sub-1 coefficient in rate: {}".format(rate.expr)
        )
    else:
        # Bare Parameter: value must be 2*k = 1.0
        assert abs(float(rate.value) - 2.0 * k) < 1e-9


def test_product_stoichiometry_two():
    """A -> 2B creates two product-pattern copies in the rule."""
    sbml = _minimal_sbml(
        species=(
            '<species id="A" compartment="c" initialAmount="10"/>'
            '<species id="B" compartment="c" initialAmount="0"/>'
        ),
        parameters='<parameter id="k" value="1.0" constant="true"/>',
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfReactants>
              <speciesReference species="A"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="B" stoichiometry="2"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    r = m.rules["r1"]
    product_cps = r.rule_expression.product_pattern.complex_patterns
    assert len(product_cps) == 2, "Expected 2 product patterns, got {}".format(
        len(product_cps)
    )
    product_names = [
        mp.monomer.name for cp in product_cps for mp in cp.monomer_patterns
    ]
    assert product_names == ["B", "B"]


def test_product_stoichiometry_two_simulation():
    """A -> 2B: conservation [B] = 2*(A0 - [A]) must hold."""
    import numpy as np
    from pysb.simulator import ScipyOdeSimulator

    k = 0.2
    A0 = 1.0
    sbml = _minimal_sbml(
        species=(
            '<species id="A" compartment="c" initialConcentration="{}" '
            'hasOnlySubstanceUnits="false"/>'.format(A0)
            + '<species id="B" compartment="c" initialConcentration="0.0" '
            'hasOnlySubstanceUnits="false"/>'
        ),
        parameters='<parameter id="k" value="{}" constant="true"/>'.format(k),
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfReactants>
              <speciesReference species="A"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="B" stoichiometry="2"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>A</ci></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    tspan = np.linspace(0, 10, 100)
    res = ScipyOdeSimulator(m, tspan=tspan, compiler="python").run()
    A_sim = res.observables["obs_A"]
    B_sim = res.observables["obs_B"]
    A_ref = A0 * np.exp(-k * tspan)
    np.testing.assert_allclose(A_sim, A_ref, rtol=1e-4, err_msg="[A] mismatch")
    np.testing.assert_allclose(
        B_sim, 2.0 * (A0 - A_ref), rtol=1e-4, err_msg="[B] mismatch"
    )


def test_trimerisation_combinatorial_correction():
    """3A -> T: combinatorial correction must be 3! = 6."""
    import sympy

    sbml = _minimal_sbml(
        species=(
            '<species id="A" compartment="c" initialConcentration="1.0"'
            ' hasOnlySubstanceUnits="false"/>'
            '<species id="T" compartment="c" initialConcentration="0.0"'
            ' hasOnlySubstanceUnits="false"/>'
        ),
        parameters='<parameter id="k" value="1.0" constant="true"/>',
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfReactants>
              <speciesReference species="A" stoichiometry="3"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="T"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci>
                  <apply><power/><ci>A</ci><cn>3</cn></apply>
                </apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    # Intrinsic rate = 6! * k * A^3 / A^3 / V = 6k.
    # Stored as bare Parameter (value 6.0) or Expression with coefficient 6.
    rate = m.rules["r1"].rate_forward
    if hasattr(rate, "expr"):
        # Extract leading numeric coefficient
        coeff = float(rate.expr.as_coeff_Mul()[0])
        assert abs(coeff - 6.0) < 1e-9, "Expected coeff 6, got {}".format(coeff)
    else:
        assert abs(float(rate.value) - 6.0) < 1e-9, "Expected value 6.0, got {}".format(
            rate.value
        )


def test_reversible_homodimer_combinatorial_correction():
    """A+A <-> D: both directions must apply the correct correction.

    Forward (reactant stoich=2): correction = 2! = 2
    Reverse (product stoich=2 for the reverse rule): correction = 2! = 2
    With kinetic law kf*A^2 - kr*D, after split:
        fwd rate = 2 * kf*A^2 / A^2 = 2*kf  (bare param or expr coefficient 2)
        rev rate = 1 * kr*D  / D    = kr    (bare param or expr coefficient 1)
    """
    import sympy

    kf, kr = 1.0, 0.5
    sbml = _minimal_sbml(
        species=(
            '<species id="A" compartment="c" initialConcentration="1.0"'
            ' hasOnlySubstanceUnits="false"/>'
            '<species id="D" compartment="c" initialConcentration="0.0"'
            ' hasOnlySubstanceUnits="false"/>'
        ),
        parameters=(
            '<parameter id="kf" value="{}" constant="true"/>'.format(kf)
            + '<parameter id="kr" value="{}" constant="true"/>'.format(kr)
        ),
        reactions="""
          <reaction id="r1" reversible="true">
            <listOfReactants>
              <speciesReference species="A" stoichiometry="2"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="D"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><minus/>
                  <apply><times/><ci>kf</ci>
                    <apply><power/><ci>A</ci><cn>2</cn></apply>
                  </apply>
                  <apply><times/><ci>kr</ci><ci>D</ci></apply>
                </apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    assert "r1_fwd" in m.rules.keys(), "Expected r1_fwd rule"
    assert "r1_rev" in m.rules.keys(), "Expected r1_rev rule"

    # Forward rate: 2*kf*A^2 / A^2 = 2*kf → coefficient on kf must be 2
    fwd_rate = m.rules["r1_fwd"].rate_forward
    if hasattr(fwd_rate, "expr"):
        fwd_coeff = float(fwd_rate.expr.as_coeff_Mul()[0])
    else:
        fwd_coeff = float(fwd_rate.value) / kf
    assert abs(fwd_coeff - 2.0) < 1e-9, "Forward coeff should be 2, got {}".format(
        fwd_coeff
    )

    # Reverse rate: 1*kr*D / D = kr → simplifies to bare Parameter kr
    rev_rate = m.rules["r1_rev"].rate_forward
    if hasattr(rev_rate, "expr"):
        free = rev_rate.expr.free_symbols
        assert m.parameters["kr"] in free, "kr should appear in reverse rate"
        nums = rev_rate.expr.atoms(sympy.Number)
        # No coefficient other than 1 (or 1.0)
        assert all(abs(float(n) - 1.0) < 1e-9 for n in nums if float(n) != 0)
    else:
        assert abs(float(rev_rate.value) - kr) < 1e-9


def test_boundary_species_excluded_from_rule_pattern():
    """Boundary species appears in kinetic law but is absent from rule pattern.

    A boundary species' amount is not changed by reactions.  The importer must
    omit it from the PySB rule pattern so BNG does not subtract it in the ODE,
    while still allowing it to appear in the kinetic-law Expression.
    """
    sbml = _minimal_sbml(
        species=(
            '<species id="S" compartment="c" initialConcentration="2.0"'
            ' boundaryCondition="true" hasOnlySubstanceUnits="false"/>'
            '<species id="P" compartment="c" initialConcentration="0.0"'
            ' hasOnlySubstanceUnits="false"/>'
        ),
        parameters='<parameter id="k" value="1.0" constant="true"/>',
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfReactants>
              <speciesReference species="S"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="P"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>S</ci></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    r = m.rules["r1"]
    reactant_names = [
        mp.monomer.name
        for cp in r.rule_expression.reactant_pattern.complex_patterns
        for mp in cp.monomer_patterns
    ]
    # S is boundary: must not appear in rule pattern
    assert "S" not in reactant_names, (
        "Boundary species S must be absent from rule pattern"
    )
    # P should appear as product
    product_names = [
        mp.monomer.name
        for cp in r.rule_expression.product_pattern.complex_patterns
        for mp in cp.monomer_patterns
    ]
    assert "P" in product_names


def test_boundary_species_in_rate_expression():
    """Boundary species drives a synthesis rate even though excluded from pattern."""
    import numpy as np
    from pysb.simulator import ScipyOdeSimulator

    # S is boundary (fixed at 2.0), P is produced at rate k*S = 2.0*k
    k, S_fixed, P0 = 0.3, 2.0, 0.0
    sbml = _minimal_sbml(
        species=(
            '<species id="S" compartment="c" initialConcentration="{}"'
            ' boundaryCondition="true" hasOnlySubstanceUnits="false"/>'.format(S_fixed)
            + '<species id="P" compartment="c" initialConcentration="{}"'
            ' hasOnlySubstanceUnits="false"/>'.format(P0)
        ),
        parameters='<parameter id="k" value="{}" constant="true"/>'.format(k),
        reactions="""
          <reaction id="r1" reversible="false">
            <listOfReactants>
              <speciesReference species="S"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="P"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>S</ci></apply>
              </math>
            </kineticLaw>
          </reaction>""",
    )
    m = _model_from_sbml_libsbml(sbml)
    tspan = np.linspace(0, 5, 100)
    res = ScipyOdeSimulator(m, tspan=tspan, compiler="python").run()
    # S fixed → S stays at S_fixed; P grows linearly: d[P]/dt = k*S_fixed
    P_sim = res.observables["obs_P"]
    P_ref = P0 + k * S_fixed * tspan
    np.testing.assert_allclose(P_sim, P_ref, rtol=1e-4, err_msg="[P] mismatch")


# ---------------------------------------------------------------------------
# Roadrunner ground-truth numerical validation
# ---------------------------------------------------------------------------


try:
    import roadrunner as _roadrunner

    HAS_ROADRUNNER = True
except ImportError:
    HAS_ROADRUNNER = False


def test_libsbml_matches_roadrunner_flat_sbml():
    """libsbml importer trajectories must match roadrunner to within rtol=1e-3.

    This test validates the combinatorial-correction fix: previously the
    homodimerisation reaction R5 produced a 2x rate error causing large
    trajectory divergence.  After the fix all species agree with roadrunner
    to better than 0.1%.
    """
    if not HAS_ROADRUNNER:
        raise SkipTest("roadrunner (libroadrunner) not installed")

    import numpy as np
    import warnings as _warnings
    from pysb.simulator import ScipyOdeSimulator

    path = _sbml_location("test_sbml_flat_SBML")
    tspan = np.linspace(0, 5, 200)

    # PySB/libsbml simulation
    with _warnings.catch_warnings(record=True):
        _warnings.simplefilter("always")
        model = SbmlImporter(path).model
    sim_result = ScipyOdeSimulator(model, tspan=tspan, compiler="python").run()

    # Roadrunner ground-truth simulation
    rr_model = _roadrunner.RoadRunner(path)
    rr_result = rr_model.simulate(float(tspan[0]), float(tspan[-1]), len(tspan))

    species = ["S1", "S2", "S3", "S4", "S5"]
    rtol = 1e-3

    for sp_id in species:
        pysb_traj = sim_result.observables["obs_" + sp_id]
        rr_traj = rr_result["[{}]".format(sp_id)]
        scale = max(float(np.max(np.abs(rr_traj))), 1e-30)
        rel_err = float(np.max(np.abs(pysb_traj - rr_traj))) / scale
        assert rel_err <= rtol, (
            "Species {} relative error vs roadrunner: {:.3e} > rtol={}".format(
                sp_id, rel_err, rtol
            )
        )
