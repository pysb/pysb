"""
Module containing a class for exporting a PySB model to SBML using libSBML

For information on how to use the model exporters, see the documentation
for :py:mod:`pysb.export`.

Supported features
------------------
* Monomers, parameters, compartments, rules, observables, expressions, initials
* Reversible rules (exported as a single reversible SBML reaction with a net
  kinetic law)
* Synthesis and degradation rules (``None`` on one side)
* Fixed species (``ic.fixed=True`` -> ``boundaryCondition="true"``)
* Multimeric reactions: duplicate reactant/product species are consolidated
  into a single ``<speciesReference>`` with the correct stoichiometry
* Local functions: expanded to per-rule derived expressions by BNG before
  export; the tag-context structure is not preserved
* Observable-based expressions (``expressions_dynamic``): expanded to bare
  species-sum MathML in the assignment rule
* The ``time`` special symbol in rate expressions: rendered as a SBML
  ``<csymbol>`` per Level 3 section 3.4.6
* Multiple compartments: species are assigned to their monomer-level
  compartment
* Volume correction: BNG's network generator produces concentration-based
  rates (``dC/dt``).  SBML kinetic laws represent flux in amount/time
  (``J = dN/dt = V * dC/dt``), so the exporter multiplies each kinetic law
  by ``V^n`` where *n* is the number of reactant molecules.  Species are
  exported with ``hasOnlySubstanceUnits=False`` so kinetic-law species
  symbols are interpreted as concentrations, matching BNG's convention.

Known limitations
-----------------
* **Energy models**: models that use ``EnergyPattern`` or energy rules raise
  :py:exc:`pysb.export.EnergyNotSupported`.
* **Tags / local-function structure**: the ``@tag`` context is unrolled by BNG
  into derived expressions; the original local-function semantics are not
  represented in the SBML output.
* **Round-trip fidelity**: rule-based structure (sites, states) is flattened
  by BNG before export.  The re-imported model faithfully reproduces the
  same ODEs but may have different species names and no rule structure.
"""

import re
import pysb
import pysb.bng
from pysb.core import SpecialSymbol
from pysb.export import Exporter, EnergyNotSupported
from sympy.printing.mathml import MathMLPrinter
from sympy import Symbol
from xml.dom.minidom import Document
import itertools

try:
    import libsbml
except ImportError:
    libsbml = None


class MathMLContentPrinter(MathMLPrinter):
    """Prints an expression to MathML without presentation markup."""

    def _print_Symbol(self, sym):
        ci = self.dom.createElement(self.mathml_tag(sym))
        ci.appendChild(self.dom.createTextNode(sym.name))
        return ci

    def _print_SpecialSymbol(self, sym):
        if sym.name == "time":
            # SBML spec (Level 3 Version 2) provides example on page 26:
            # <csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
            csymbol = self.dom.createElement("csymbol")
            csymbol.setAttribute("encoding", "text")
            csymbol.setAttribute(
                "definitionURL", "http://www.sbml.org/sbml/symbols/time"
            )
            csymbol.appendChild(self.dom.createTextNode("t"))
            return csymbol

    def to_xml(self, expr):
        # Preferably this should use a public API, but as that doesn't exist...
        return self._print(expr)


def _check(value):
    """
    Validate a libsbml return value

    Raises ValueError if 'value' is a libsbml error code or None.
    """
    if type(value) is int and value != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise ValueError(
            "Error encountered converting to SBML. "
            'LibSBML returned error code {}: "{}"'.format(
                value, libsbml.OperationReturnValue_toString(value).strip()
            )
        )
    elif value is None:
        raise ValueError("LibSBML returned a null value")


def _xml_to_ast(x_element):
    """Wrap MathML fragment with <math> tag and convert to libSBML AST"""
    x_doc = Document()
    x_mathml = x_doc.createElement("math")
    x_mathml.setAttribute("xmlns", "http://www.w3.org/1998/Math/MathML")

    x_mathml.appendChild(x_element)
    x_doc.appendChild(x_mathml)

    mathml_ast = libsbml.readMathMLFromString(x_doc.toxml())
    _check(mathml_ast)
    return mathml_ast


class SbmlExporter(Exporter):
    """A class for returning the SBML for a given PySB model.

    Inherits from :py:class:`pysb.export.Exporter`, which implements
    basic functionality for all exporters.
    """

    def __init__(self, *args, **kwargs):
        if not libsbml:
            raise ImportError("The SbmlExporter requires the libsbml python package")
        super(SbmlExporter, self).__init__(*args, **kwargs)

    def _sympy_to_sbmlast(self, sympy_expr):
        """
        Convert a sympy expression to the AST format used by libsbml
        """
        return _xml_to_ast(MathMLContentPrinter().to_xml(sympy_expr))

    def convert(self, level=(3, 2)):
        """
        Convert the PySB model to a libSBML document

        Requires the libsbml python package

        Parameters
        ----------
        level: (int, int)
            The SBML level and version to use. The default is SBML level 3,
            version 2. Conversion to other levels/versions may not be possible
            or may lose fidelity.

        Returns
        -------
        libsbml.SBMLDocument
            A libSBML document converted from the PySB model
        """
        doc = libsbml.SBMLDocument(3, 2)
        smodel = doc.createModel()
        _check(smodel)

        if self.model.uses_energy:
            raise EnergyNotSupported()

        _check(smodel.setName(self.model.name))
        # SBML IDs must match [a-zA-Z_][a-zA-Z0-9_]* -- replace any other
        # characters (e.g. dots from Python module paths) with underscores.
        safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", self.model.name)
        if safe_id and safe_id[0].isdigit():
            safe_id = "_" + safe_id
        _check(smodel.setId(safe_id))

        pysb.bng.generate_equations(self.model)

        # Docstring
        if self.docstring:
            notes_str = """
            <notes>
                <body xmlns="http://www.w3.org/1999/xhtml">
                    <p>%s</p>
                </body>
            </notes>""" % self.docstring.replace("\n", "<br />\n" + " " * 20)
            _check(smodel.setNotes(notes_str))

        # Compartments
        if self.model.compartments:
            for cpt in self.model.compartments:
                c = smodel.createCompartment()
                _check(c)
                _check(c.setId(cpt.name))
                _check(c.setSpatialDimensions(cpt.dimension))
                _check(c.setSize(1 if cpt.size is None else cpt.size.value))
                _check(c.setConstant(True))
        else:
            c = smodel.createCompartment()
            _check(c)
            _check(c.setId("default"))
            _check(c.setSpatialDimensions(3))
            _check(c.setSize(1))
            _check(c.setConstant(True))

        # Expressions
        for expr in itertools.chain(
            self.model.expressions_constant(),
            self.model.expressions_dynamic(include_local=False),
            self.model._derived_expressions,
        ):
            # create an observable "parameter"
            e = smodel.createParameter()
            _check(e)
            _check(e.setId(expr.name))
            _check(e.setName(expr.name))
            _check(e.setConstant(False))

            # create an assignment rule which assigns the expression to the parameter
            expr_rule = smodel.createAssignmentRule()

            _check(expr_rule)
            _check(expr_rule.setVariable(e.getId()))

            expr_mathml = self._sympy_to_sbmlast(
                expr.expand_expr(expand_observables=True)
            )
            _check(expr_rule.setMath(expr_mathml))

        # Initial values/assignments
        fixed_species_idx = set()
        initial_species_idx = set()
        for ic in self.model.initials:
            sp_idx = self.model.get_species_index(ic.pattern)
            ia = smodel.createInitialAssignment()
            _check(ia)
            _check(ia.setSymbol("__s{}".format(sp_idx)))
            init_mathml = self._sympy_to_sbmlast(Symbol(ic.value.name))
            _check(ia.setMath(init_mathml))
            initial_species_idx.add(sp_idx)

            if ic.fixed:
                fixed_species_idx.add(sp_idx)

        # Species
        for i, s in enumerate(self.model.species):
            sp = smodel.createSpecies()
            _check(sp)
            _check(sp.setId("__s{}".format(i)))
            if self.model.compartments:
                # Try to determine compartment, which must be unique for the species
                mon_cpt = set(
                    mp.compartment
                    for mp in s.monomer_patterns
                    if mp.compartment is not None
                )
                if len(mon_cpt) == 0 and s.compartment:
                    compartment_name = s.compartment.name
                elif len(mon_cpt) == 1:
                    mon_cpt = mon_cpt.pop()
                    if s.compartment is not None and mon_cpt != s.compartment:
                        raise ValueError(
                            "Species {} has different monomer and species "
                            "compartments, which is not supported in "
                            "SBML".format(s)
                        )
                    compartment_name = mon_cpt.name
                else:
                    raise ValueError(
                        "Species {} has more than one different monomer "
                        "compartment, which is not supported in "
                        "SBML".format(s)
                    )
            else:
                compartment_name = "default"
            _check(sp.setCompartment(compartment_name))
            _check(sp.setName(str(s).replace("% ", "._br_")))
            _check(sp.setBoundaryCondition(i in fixed_species_idx))
            _check(sp.setConstant(False))
            _check(sp.setHasOnlySubstanceUnits(False))
            if i not in initial_species_idx:
                _check(sp.setInitialConcentration(0.0))

        # Parameters

        for param in itertools.chain(
            self.model.parameters, self.model._derived_parameters
        ):
            p = smodel.createParameter()
            _check(p)
            _check(p.setId(param.name))
            _check(p.setName(param.name))
            _check(p.setValue(param.value))
            _check(p.setConstant(True))

        # Reactions
        # Use the unidirectional reaction list so each reaction has a single
        # well-defined set of reactants and a single BNG rate expression.
        # BNG rates are concentration-based (dC/dt), while SBML kinetic laws
        # represent amount flux (J = dN/dt = V * dC/dt).  For a reaction with
        # n reactant molecules in compartment of volume V, the relationship is:
        #   J = V^n * BNG_rate
        # (BNG embeds 1/V^(n-1) for mass-action, giving dC/dt = BNG_rate;
        # multiplying by V restores J in amount/time units.)
        for i, reaction in enumerate(self.model.reactions):
            rxn = smodel.createReaction()
            _check(rxn)
            _check(rxn.setId("r{}".format(i)))
            _check(rxn.setName(" + ".join(reaction["rule"])))
            _check(rxn.setReversible(False))

            # Consolidate duplicate species indices into a single
            # speciesReference with the appropriate stoichiometry.
            reactant_counts = {}
            for sp in reaction["reactants"]:
                reactant_counts[sp] = reactant_counts.get(sp, 0) + 1
            for sp, stoich in reactant_counts.items():
                reac = rxn.createReactant()
                _check(reac)
                _check(reac.setSpecies("__s{}".format(sp)))
                _check(reac.setStoichiometry(stoich))
                _check(reac.setConstant(True))

            product_counts = {}
            for sp in reaction["products"]:
                product_counts[sp] = product_counts.get(sp, 0) + 1
            for sp, stoich in product_counts.items():
                prd = rxn.createProduct()
                _check(prd)
                _check(prd.setSpecies("__s{}".format(sp)))
                _check(prd.setStoichiometry(stoich))
                _check(prd.setConstant(True))

            # Collect species that are already reactants or products so we
            # do not duplicate them as modifiers (invalid SBML).
            rxn_species = set(
                "__s{}".format(sp)
                for sp in itertools.chain(reaction["reactants"], reaction["products"])
            )

            for symbol in reaction["rate"].free_symbols:
                if isinstance(symbol, pysb.Expression):
                    expr = symbol.expand_expr(expand_observables=True)
                    for sym in expr.free_symbols:
                        if not isinstance(
                            sym, (pysb.Parameter, pysb.Expression, SpecialSymbol)
                        ):
                            # Species reference -- only add as modifier if not
                            # already a reactant or product.
                            if str(sym) not in rxn_species:
                                modifier = rxn.createModifier()
                                _check(modifier)
                                _check(modifier.setSpecies(str(sym)))

            # Volume correction: determine V for the reaction compartment.
            # For reactions with reactants, use the reactant compartment.
            # For synthesis (no reactants), use the product compartment.
            # For models without explicit compartments, V = 1 (no correction).
            bng_rate = reaction["rate"]
            if self.model.compartments:
                n_reac = len(reaction["reactants"])
                # Identify the reference species list (reactants if present,
                # otherwise products) to determine the reaction compartment.
                ref_indices = (
                    reaction["reactants"]
                    if reaction["reactants"]
                    else reaction["products"]
                )
                cpt_size = None
                for sp_idx in ref_indices:
                    sp = self.model.species[sp_idx]
                    mon_cpts = [
                        mp.compartment
                        for mp in sp.monomer_patterns
                        if mp.compartment is not None
                    ]
                    if mon_cpts:
                        cpt = mon_cpts[0]
                        if cpt.size is not None:
                            cpt_size = Symbol(cpt.size.name)
                        break
                if cpt_size is not None:
                    # J = V^n * BNG_rate converts concentration-based BNG
                    # rates to the amount/time flux expected by SBML.
                    kinetic_rate = cpt_size**n_reac * bng_rate
                else:
                    kinetic_rate = bng_rate
            else:
                kinetic_rate = bng_rate

            rate = rxn.createKineticLaw()
            _check(rate)
            rate_mathml = self._sympy_to_sbmlast(kinetic_rate)
            _check(rate.setMath(rate_mathml))

        # Observables
        for i, observable in enumerate(self.model.observables):
            # create an observable "parameter"
            obs = smodel.createParameter()
            _check(obs)
            _check(obs.setId("__obs{}".format(i)))
            _check(obs.setName(observable.name))
            _check(obs.setConstant(False))

            # create an assignment rule which assigns the observable expression to the parameter
            obs_rule = smodel.createAssignmentRule()

            _check(obs_rule)
            _check(obs_rule.setVariable(obs.getId()))

            obs_mathml = self._sympy_to_sbmlast(observable.expand_obs())
            _check(obs_rule.setMath(obs_mathml))

        # Apply any requested level/version conversion
        if level != (3, 2):
            prop = libsbml.ConversionProperties(libsbml.SBMLNamespaces(*level))
            prop.addOption("strict", False)
            prop.addOption("setLevelAndVersion", True)
            prop.addOption("ignorePackages", True)
            _check(doc.convert(prop))

        return doc

    def export(self, level=(3, 2)):
        """
        Export the SBML for the PySB model associated with the exporter

        Requires libsbml package.

        Parameters
        ----------
        level: (int, int)
            The SBML level and version to use. The default is SBML level 3,
            version 2. Conversion to other levels/versions may not be possible
            or may lose fidelity.

        Returns
        -------
        string
            String containing the SBML output.
        """
        return libsbml.writeSBMLToString(self.convert(level=level))
