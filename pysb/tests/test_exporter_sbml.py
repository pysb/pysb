"""
Unit tests for the libsbml-based SBML exporter (pysb.export.sbml.SbmlExporter).
"""

import re
from collections import Counter

from nose.plugins.skip import SkipTest
from nose.tools import assert_raises

from pysb.testing import with_model
from pysb import (
    Monomer,
    Parameter,
    Initial,
    Rule,
    Observable,
    Expression,
    time,
)
from pysb.core import EnergyPattern
from pysb.bng import generate_equations
from pysb.export import EnergyNotSupported

try:
    import libsbml

    HAS_LIBSBML = True
except ImportError:
    HAS_LIBSBML = False

if not HAS_LIBSBML:
    raise SkipTest("libsbml not available")

from pysb.export.sbml import SbmlExporter


def _get_sbml_model(pysb_model):
    """Return the libsbml Model object for *pysb_model*."""
    return SbmlExporter(pysb_model).convert().getModel()


def _sbml_string(pysb_model):
    """Return the SBML XML string for *pysb_model*."""
    return SbmlExporter(pysb_model).export()


# Model ID


@with_model
def test_model_id_simple_name():
    """A model whose name contains no special characters is exported as-is."""
    Monomer("A")
    Parameter("kdeg", 1.0)
    Initial(A(), kdeg)
    Rule("deg", A() >> None, kdeg)

    # Rename the model after construction to avoid the auto-generated name.
    model.name = "MySimpleModel"

    smodel = _get_sbml_model(model)
    assert smodel.getId() == "MySimpleModel", (
        'Expected SBML model id "MySimpleModel", got "{}"'.format(smodel.getId())
    )


@with_model
def test_model_id_dotted_name():
    """Dots (from Python module paths) are replaced with underscores in the id."""
    Monomer("A")
    Parameter("kdeg", 1.0)
    Initial(A(), kdeg)
    Rule("deg", A() >> None, kdeg)

    model.name = "pysb.examples.robertson"

    smodel = _get_sbml_model(model)
    model_id = smodel.getId()
    # Dots must be gone; only word characters are allowed in SBML IDs.
    assert re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", model_id), (
        'SBML model id "{}" is not a valid SBML identifier'.format(model_id)
    )
    assert model_id == "pysb_examples_robertson", (
        'Expected "pysb_examples_robertson", got "{}"'.format(model_id)
    )


@with_model
def test_model_id_roundtrip():
    """The model id written by the exporter can be read back by libsbml."""
    Monomer("A")
    Parameter("kdeg", 1.0)
    Initial(A(), kdeg)
    Rule("deg", A() >> None, kdeg)

    model.name = "RoundTripModel"

    sbml_str = _sbml_string(model)
    doc2 = libsbml.readSBMLFromString(sbml_str)
    assert doc2.getModel().getId() == "RoundTripModel"


# Stoichiometry consolidation


@with_model
def test_stoichiometry_no_duplicate_reactants():
    """A homodimerisation reaction must not produce duplicate speciesReference elements."""
    Monomer("A")
    Monomer("AA")
    Parameter("kf", 1e-3)
    Parameter("A_0", 100.0)
    Initial(A(), A_0)
    Rule("homodimer", A() + A() >> AA(), kf)

    smodel = _get_sbml_model(model)

    for rxn_idx in range(smodel.getNumReactions()):
        rxn = smodel.getReaction(rxn_idx)
        reactant_species = [
            rxn.getReactant(j).getSpecies() for j in range(rxn.getNumReactants())
        ]
        counts = Counter(reactant_species)
        duplicates = {sp: n for sp, n in counts.items() if n > 1}
        assert not duplicates, (
            "Reaction {} has duplicate reactant speciesReferences: {}".format(
                rxn.getId(), duplicates
            )
        )


@with_model
def test_stoichiometry_value_homodimerisation():
    """A + A -> AA should export A as a reactant with stoichiometry 2."""
    Monomer("A")
    Monomer("AA")
    Parameter("kf", 1e-3)
    Parameter("A_0", 100.0)
    Initial(A(), A_0)
    Rule("homodimer", A() + A() >> AA(), kf)

    generate_equations(model)

    # Find the species index for A(). Species are ComplexPatterns; match by str.
    a_idx = next(i for i, s in enumerate(model.species) if str(s) == "A()")

    smodel = _get_sbml_model(model)

    found = False
    for rxn_idx in range(smodel.getNumReactions()):
        rxn = smodel.getReaction(rxn_idx)
        for j in range(rxn.getNumReactants()):
            sr = rxn.getReactant(j)
            if sr.getSpecies() == "__s{}".format(a_idx):
                assert sr.getStoichiometry() == 2.0, (
                    "Expected stoichiometry 2 for __s{} in {}, got {}".format(
                        a_idx, rxn.getId(), sr.getStoichiometry()
                    )
                )
                found = True
    assert found, "Species __s{} not found as reactant in any reaction".format(a_idx)


@with_model
def test_stoichiometry_no_duplicate_products():
    """A dissociation reaction producing two identical species must use stoichiometry=2."""
    Monomer("A")
    Monomer("AA")
    Parameter("kr", 1e-1)
    Parameter("AA_0", 50.0)
    Initial(AA(), AA_0)
    Rule("dissoc", AA() >> A() + A(), kr)

    generate_equations(model)
    a_idx = next(i for i, s in enumerate(model.species) if str(s) == "A()")
    smodel = _get_sbml_model(model)

    for rxn_idx in range(smodel.getNumReactions()):
        rxn = smodel.getReaction(rxn_idx)
        product_species = [
            rxn.getProduct(j).getSpecies() for j in range(rxn.getNumProducts())
        ]
        counts = Counter(product_species)
        duplicates = {sp: n for sp, n in counts.items() if n > 1}
        assert not duplicates, (
            "Reaction {} has duplicate product speciesReferences: {}".format(
                rxn.getId(), duplicates
            )
        )
        # Also verify the consolidated stoichiometry
        for j in range(rxn.getNumProducts()):
            pr = rxn.getProduct(j)
            if pr.getSpecies() == "__s{}".format(a_idx):
                assert pr.getStoichiometry() == 2.0, (
                    "Expected stoichiometry 2 for __s{} product, got {}".format(
                        a_idx, pr.getStoichiometry()
                    )
                )


# Modifiers must not duplicate reactants, products, or special symbols


@with_model
def test_no_reactant_as_modifier():
    """Species listed as reactants must not also appear as modifiers."""
    Monomer("S")
    Monomer("E")
    Monomer("P")
    Parameter("kcat", 1.0)
    Parameter("Km", 10.0)
    Parameter("S_0", 100.0)
    Parameter("E_0", 1.0)
    Observable("obs_S", S())
    Observable("obs_E", E())
    Expression("rate_expr", obs_S * obs_E * kcat / (obs_S + Km))
    Initial(S(), S_0)
    Initial(E(), E_0)
    Rule("conversion", S() >> P(), rate_expr)

    smodel = _get_sbml_model(model)

    for rxn_idx in range(smodel.getNumReactions()):
        rxn = smodel.getReaction(rxn_idx)
        reactant_ids = {
            rxn.getReactant(j).getSpecies() for j in range(rxn.getNumReactants())
        }
        product_ids = {
            rxn.getProduct(j).getSpecies() for j in range(rxn.getNumProducts())
        }
        rxn_species = reactant_ids | product_ids

        for mod_idx in range(rxn.getNumModifiers()):
            mod_id = rxn.getModifier(mod_idx).getSpecies()
            assert mod_id not in rxn_species, (
                "Reaction {}: species {} is listed as both a "
                "reactant/product AND a modifier".format(rxn.getId(), mod_id)
            )


@with_model
def test_no_product_as_modifier():
    """Species listed as products must not also appear as modifiers."""
    Monomer("A")
    Monomer("B")
    Observable("obs_B", B())
    Parameter("krate", 1.0)
    Parameter("A_0", 50.0)
    Expression("rate_expr", obs_B * krate)
    Initial(A(), A_0)
    Rule("make_B", A() >> B(), rate_expr)

    smodel = _get_sbml_model(model)

    for rxn_idx in range(smodel.getNumReactions()):
        rxn = smodel.getReaction(rxn_idx)
        reactant_ids = {
            rxn.getReactant(j).getSpecies() for j in range(rxn.getNumReactants())
        }
        product_ids = {
            rxn.getProduct(j).getSpecies() for j in range(rxn.getNumProducts())
        }
        rxn_species = reactant_ids | product_ids

        for mod_idx in range(rxn.getNumModifiers()):
            mod_id = rxn.getModifier(mod_idx).getSpecies()
            assert mod_id not in rxn_species, (
                "Reaction {}: species {} is listed as both a "
                "reactant/product AND a modifier".format(rxn.getId(), mod_id)
            )


@with_model
def test_time_not_added_as_modifier():
    """The PySB `time` symbol in a rate expression must not appear as a modifier."""
    from sympy import exp

    Monomer("A")
    Parameter("kA", 1.0)
    Parameter("A_0", 100.0)
    # Rate expression that explicitly uses the `time` SpecialSymbol.
    Expression("rate_expr", kA * exp(-time))
    Initial(A(), A_0)
    Rule("decay", A() >> None, rate_expr)

    smodel = _get_sbml_model(model)

    for rxn_idx in range(smodel.getNumReactions()):
        rxn = smodel.getReaction(rxn_idx)
        for mod_idx in range(rxn.getNumModifiers()):
            mod_id = rxn.getModifier(mod_idx).getSpecies()
            assert mod_id != "time", (
                'Reaction {}: "time" was incorrectly added as a '
                "modifierSpeciesReference".format(rxn.getId())
            )


# Fixed species, docstring notes, and level conversion


@with_model
def test_fixed_species_boundary_condition():
    """A fixed initial condition is exported as boundaryCondition=true."""
    Monomer("A")
    Parameter("A_0", 100.0)
    Parameter("kdeg", 1.0)
    Initial(A(), A_0, fixed=True)
    Rule("deg", A() >> None, kdeg)

    smodel = _get_sbml_model(model)

    a_species = next(
        smodel.getSpecies(i)
        for i in range(smodel.getNumSpecies())
        if smodel.getSpecies(i).getName() == "A()"
    )
    assert a_species.getBoundaryCondition(), (
        "Fixed species should have boundaryCondition=true"
    )


@with_model
def test_docstring_exported_as_notes():
    """A docstring passed to the exporter is written as SBML notes."""
    Monomer("A")
    Parameter("kdeg", 1.0)
    Initial(A(), kdeg)
    Rule("deg", A() >> None, kdeg)

    doc = SbmlExporter(model, docstring="Test model description.").convert()
    notes = doc.getModel().getNotesString()
    assert "Test model description." in notes


@with_model
def test_level_conversion():
    """Requesting SBML Level 2 produces a valid document at that level."""
    Monomer("A")
    Parameter("kdeg", 1.0)
    Initial(A(), kdeg)
    Rule("deg", A() >> None, kdeg)

    doc = SbmlExporter(model).convert(level=(2, 4))
    assert doc.getLevel() == 2


# Energy models


@with_model
def test_energy_model_raises():
    """Exporting an energy model must raise EnergyNotSupported."""
    Monomer("A")
    Parameter("G_A", 1.0)
    EnergyPattern("ep_A", A(), G_A)

    assert_raises(EnergyNotSupported, _sbml_string, model)


@with_model
def test_catalyst_reaction_kinetic_law_volume_exponent():
    """Catalyst reaction B+B->C+B in a non-unit compartment: kinetic law must
    use V^1 (net-consumed = 1), not V^2 (raw reactant count = 2).

    For Robertson's second reaction (B+B->C+B, BNG rate = k*[B]^2), the
    SBML kinetic law J = V^n_net * k*[B]^2 where n_net = 1 (one B is
    consumed net).  Previously the exporter used len(reactants)=2, writing
    V^2*k*[B]^2 which inflated J by V and broke round-trip accuracy for
    non-unit compartments.
    """
    from pysb import Compartment
    from pysb.bng import generate_equations

    Compartment("cell", dimension=3, size=Parameter("Vcell", 2.0))
    Monomer("B")
    Monomer("C")
    Parameter("k2", 3e7)
    Initial(B() ** cell, Parameter("B_0", 1e-3))
    Initial(C() ** cell, Parameter("C_0", 0.0))
    Rule("BB_to_BC", B() ** cell + B() ** cell >> C() ** cell + B() ** cell, k2)
    Observable("obs_B", B() ** cell)
    generate_equations(model)

    sbml_model = _get_sbml_model(model)
    # Find the reaction in the exported SBML
    assert sbml_model.getNumReactions() == 1
    rxn = sbml_model.getReaction(0)
    math_str = libsbml.formulaToL3String(rxn.getKineticLaw().getMath())

    # The kinetic law must contain Vcell^1 (i.e. "Vcell" appears exactly once
    # as a factor), not Vcell^2.  We check that the string does NOT contain
    # "Vcell^2" or "Vcell * Vcell".
    assert "Vcell^2" not in math_str, (
        "Kinetic law should use V^1 for catalyst reaction, got: {}".format(math_str)
    )
    assert "Vcell * Vcell" not in math_str, (
        "Kinetic law should use V^1 for catalyst reaction, got: {}".format(math_str)
    )
    # And Vcell must appear at least once (volume correction is applied)
    assert "Vcell" in math_str, (
        "Kinetic law must include compartment volume, got: {}".format(math_str)
    )
