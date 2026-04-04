"""
Import SBML models into PySB.

Two backends are provided:

**libsbml (default)**
    :class:`SbmlImporter` reads an SBML file via the ``python-libsbml``
    package and builds a PySB model directly.  No external programs are
    required.  Use :func:`model_from_sbml` (``use_libsbml=True``, the
    default).

**Legacy sbmlTranslator / Atomizer**
    The BioNetGen ``sbmlTranslator`` binary (also known as *Atomizer*) can
    attempt to infer higher-level rule-based structure from SBML.  Pass
    ``use_libsbml=False`` to :func:`model_from_sbml` to use this path.
    Atomizer must be installed separately (see :func:`sbml_translator`).

SBML level and version support (libsbml importer)
--------------------------------------------------

The libsbml importer supports **SBML Level 2 Version 1–5** and
**Level 3 Version 1–2**, which covers the vast majority of models in
public repositories such as BioModels.

**Level 1** SBML (versions 1–2) is read by libsbml and converted to an
equivalent internal representation, so most Level 1 files will import
correctly.  However Level 1 lacks several features (e.g. MathML,
compartments, assignment rules, function definitions) and has not been
validated as thoroughly as Level 2/3.

SBML feature support (libsbml importer)
----------------------------------------

The table below summarises which SBML constructs are handled and how they
map to PySB components.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - SBML construct
     - PySB representation
   * - ``<species>``
     - :class:`~pysb.core.Monomer` with no sites +
       :class:`~pysb.core.Observable` ``obs_<id>`` for use in rate laws
   * - ``<reaction>`` (irreversible) with ``<kineticLaw>``
     - :class:`~pysb.core.Rule` with a function-based
       :class:`~pysb.core.Expression` rate equal to the kinetic law
       **divided by the product of reactant observables and by the reaction
       compartment volume** (so that PySB/BNG's implicit reactant-count
       multiplication yields the correct concentration ODE flux,
       ``dC/dt = J/V``).  A combinatorial correction (``n!`` for each
       species appearing with stoichiometry *n*) is applied to cancel
       BNG's internal symmetry-factor division for homo-multimers.
       When reactants and products reside in different compartments
       (transport), two rules are generated: a degradation rule for the
       source compartment and a synthesis rule for the destination
       compartment, so that each species ODE is divided by its own
       compartment volume.
   * - ``<reaction>`` (reversible, ``reversible="true"``) with ``<kineticLaw>``
     - Two non-reversible :class:`~pysb.core.Rule` objects (``<id>_fwd``
       and ``<id>_rev``).  The net-flux kinetic law is split into positive
       and negative additive terms; each half is divided by the product of
       the corresponding species observables, by the compartment volume,
       and by the combinatorial correction.  If the split fails a warning
       is issued and only the forward rule is created.  Cross-compartment
       reactions produce up to four rules (``_fwd_deg``, ``_fwd_prod``,
       ``_rev_deg``, ``_rev_prod``) for the same reason as the irreversible
       case above.
   * - ``<parameter>``
     - :class:`~pysb.core.Parameter` (``nonnegative=False`` so negative
       values such as reversal potentials are accepted)
   * - ``<compartment>``
     - :class:`~pysb.core.Compartment` + a size
       :class:`~pysb.core.Parameter` named ``<id>_size``
   * - ``<assignmentRule>``
     - :class:`~pysb.core.Expression`
   * - ``<rateRule>`` (ODE-only models)
     - :class:`~pysb.core.Monomer` ``X()`` + production rule
       ``None >> X()`` whose rate expression equals the full RHS, encoding
       ``dX/dt = f(...)`` faithfully.  Concentrations may go negative
       (e.g. membrane voltage), which :class:`~pysb.simulator.ScipyOdeSimulator`
       handles correctly.
   * - ``<initialAssignment>``
     - Updates the species initial-condition parameter value
   * - ``<functionDefinition>``
     - Parsed to a Python callable used during kinetic-law function parsing
   * - ``csymbol`` ``time``
     - Maps to :data:`pysb.core.time`, PySB's simulation-time symbol
   * - ``boundaryCondition="true"``
     - :class:`~pysb.core.Initial` with ``fixed=True``; the species is
       **excluded from all rule patterns** so BNG does not include it in
       ODEs, but it may still appear in kinetic-law expressions
   * - Integer stoichiometry *n* > 1
     - *n* copies of the reactant/product :class:`~pysb.core.ComplexPattern`
       in the rule, plus the combinatorial correction described above

Known limitations
-----------------

* **Flat import only.** All species are created as site-less Monomers; no
  binding-site structure is inferred.  For structure inference (atomisation)
  use ``model_from_sbml(filename, use_libsbml=False, atomize=True)``.

* **Algebraic rules are not supported** and raise :class:`SbmlImportError`
  (or warn with ``force=True``).  Algebraic rules define implicit
  constraints ``0 = f(...)`` that require a DAE solver, which PySB does
  not provide.

* **SBML Events are not supported** and raise :class:`SbmlImportError`
  (or warn with ``force=True``).  Events describe triggered discontinuities
  (e.g. stimulus pulses) that have no rule-based analogue in PySB.

* **Non-integer stoichiometry** is truncated to the nearest integer and a
  :mod:`warnings` warning is issued.  Use kinetic-law expressions to
  represent fractional stoichiometries.

* **Rate-rule variables placed in first compartment.** When a model has
  compartments, rate-rule variables (which are SBML parameters, not species)
  are assigned to the first compartment to satisfy PySB's concreteness
  requirement.  This is a formal annotation that does not affect the ODEs.

* **Stochastic simulation of rate-rule models.** ``None >> X()`` rules with
  expressions that may be negative have no valid stochastic interpretation.
  Use :class:`~pysb.simulator.ScipyOdeSimulator` for ODE integration.

* **Unit definitions** are parsed but ignored; all quantities are treated as
  dimensionless.

* **Dynamic compartments** (compartments whose size changes via a rule) raise
  :class:`SbmlImportError` (or warn with ``force=True``); the compartment
  retains its initial size.

* **Non-separable reversible kinetic laws**: when a reversible reaction's
  kinetic law cannot be split into distinct positive (forward) and negative
  (reverse) additive terms (e.g. a single Michaelis-Menten-like fraction that
  spans both directions), :func:`_split_reversible_rate` returns
  ``(None, None)`` and a warning is issued.  Only the forward rule is created
  and the reverse direction is absent from the model.

* **Reaction ``fast`` attribute**: ``fast="true"`` (quasi-steady-state,
  deprecated in SBML Level 3 Version 2) issues a :mod:`warnings` warning and
  is otherwise ignored.

* **Local kinetic-law parameters**: parameters declared inside a
  ``<kineticLaw>`` element that shadow global parameters of the same name are
  not added to the PySB model.  A :mod:`warnings` warning is issued listing
  the affected parameter IDs; the global parameter is used in its place.

* **Unnamed reactions**: reactions with no ``id`` attribute are assigned the
  fallback name ``r0``, ``r1``, …, ``rN``.

* **Zero stoichiometry**: a stoichiometry value of 0 (unusual but valid in
  SBML) produces zero pattern copies and is effectively a no-op for that
  species reference.

* **Multiple rate rules on the same variable** are not supported; the second
  rule encountered for a given variable raises :class:`SbmlImportError` (or
  warns with ``force=True``) because duplicate PySB rules are not valid.
"""

from pysb.importers.bngl import model_from_bngl
from pysb.builder import Builder
from pysb.core import (
    MonomerPattern,
    ComplexPattern,
    RuleExpression,
    ReactionPattern,
    time as pysb_time,
)
from pysb.bng import parse_bngl_expr
import pysb.logging
import pysb.pathfinder as pf
import subprocess
import os
import math
import re
import tempfile
import shutil
import sympy
import warnings
from urllib.request import urlretrieve
from pysb.logging import get_logger, EXTENDED_DEBUG

try:
    import libsbml
except ImportError:
    libsbml = None

BIOMODELS_REGEX = re.compile(r"(BIOMD|MODEL)[0-9]{10}")
BIOMODELS_URLS = {
    "ebi": "http://www.ebi.ac.uk/biomodels-main/download?mid={}",
    "caltech": "http://biomodels.caltech.edu/download?mid={}",
}


class SbmlTranslationError(Exception):
    pass


class SbmlImportError(Exception):
    pass


def _sanitize_id(sbml_id):
    """Convert an SBML ID to a valid Python/PySB identifier"""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", sbml_id)
    if name and name[0].isdigit():
        name = "s_" + name
    return name or "unnamed"


_SBML_MATH_FUNCTIONS = {
    "ln": sympy.log,
    "log": sympy.log,
    "exp": sympy.exp,
    "sqrt": sympy.sqrt,
    "abs": sympy.Abs,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "floor": sympy.floor,
    "ceiling": sympy.ceiling,
    "pow": sympy.Pow,
    "root": sympy.sqrt,
    "factorial": sympy.factorial,
    "tanh": sympy.tanh,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
}


class SbmlImporter(Builder):
    """Import a flat SBML model using libsbml.

    Reads an SBML file and constructs a PySB model.  The importer is a
    :class:`~pysb.builder.Builder` subclass, so the resulting model is
    available as ``importer.model`` after construction.

    Parameters
    ----------
    filename : str
        Path to the SBML file (any SBML Level/Version supported by libsbml).
    force : bool, optional
        If False (default), raises :class:`SbmlImportError` when an
        unsupported construct is encountered.  If True, issues a
        :mod:`warnings` warning and continues, producing a partial model.

    Raises
    ------
    ImportError
        If the ``libsbml`` Python package is not installed.
    SbmlImportError
        If the SBML file contains fatal errors or an unsupported construct
        is encountered (and ``force=False``).

    Notes
    -----
    For most use cases the convenience functions :func:`model_from_sbml`,
    or :func:`model_from_biomodels` are easier to use than instantiating
    :class:`SbmlImporter` directly.

    **SBML ID sanitisation**: SBML IDs may contain characters that are
    invalid in Python identifiers (hyphens, dots, etc.).  These are replaced
    with underscores by :func:`_sanitize_id`.  IDs that start with a digit
    are prefixed with ``s_``.

    **Species/observables**: each SBML ``<species>`` element becomes a
    :class:`~pysb.core.Monomer` named after its SBML ID (sanitised) and a
    corresponding :class:`~pysb.core.Observable` named ``obs_<id>``.  The
    observable is used in kinetic-law and assignment-rule expressions so that
    species amounts appear naturally in rate expressions.  Initial conditions
    are created as ``<id>_0`` :class:`~pysb.core.Parameter` objects.

    **Reactions**: each SBML ``<reaction>`` becomes a
    :class:`~pysb.core.Rule`.  The kinetic law is parsed and stored as a
    function-based :class:`~pysb.core.Expression`; BNG/PySB then use the
    expression value directly as the ODE flux (no additional multiplication
    by species counts).

    **Parameters**: SBML ``<parameter>`` elements become PySB
    :class:`~pysb.core.Parameter` objects with ``nonnegative=False`` so that
    negative values (e.g. reversal potentials in electrophysiology models)
    are accepted without error.

    **Compartments**: each ``<compartment>`` becomes both a PySB
    :class:`~pysb.core.Compartment` and a size
    :class:`~pysb.core.Parameter` named ``<id>_size``.  The size parameter
    is used inside kinetic-law expressions wherever the compartment ID
    appears (consistent with the SBML convention of multiplying rates by
    compartment volume).

    **Assignment rules**: ``<assignmentRule>`` elements create PySB
    :class:`~pysb.core.Expression` objects whose value tracks the rule RHS at
    every time point.  The corresponding SBML parameter is *not* added to the
    model as a plain ``Parameter``.

    **Rate rules**: ``<rateRule>`` elements (ODE-only models) are
    represented as synthesis rules ``None >> X()`` with a function-based rate
    equal to the full RHS.  Because PySB/BNG use the expression as the
    direct ODE flux, this faithfully encodes ``dX/dt = f(...)``.  Negative
    flux values are valid (e.g. membrane voltage) and are handled correctly by
    :class:`~pysb.simulator.ScipyOdeSimulator`.

    **FunctionDefinitions**: SBML ``<functionDefinition>`` elements are
    parsed into Python callables (via :mod:`sympy` substitution) and made
    available when evaluating kinetic-law functions.

    **Initial assignments**: ``<initialAssignment>`` elements override
    species initial values.  This is the mechanism used by PySB's own SBML
    exporter, so round-trip import/export is supported.

    Examples
    --------
    >>> from pysb.importers.sbml import SbmlImporter
    >>> imp = SbmlImporter('my_model.xml')        # doctest: +SKIP
    >>> model = imp.model                         # doctest: +SKIP
    >>> print(model.monomers)                     # doctest: +SKIP
    """

    def __init__(self, filename_or_string, force=False):
        super().__init__()

        if libsbml is None:
            raise ImportError("The SbmlImporter requires the libsbml python package")

        reader = libsbml.SBMLReader()

        # Accept either a path to an SBML file or a raw SBML string/bytes so
        # that callers (especially tests) can avoid writing temporary files.
        if isinstance(filename_or_string, bytes):
            sbml_string = filename_or_string.decode("utf-8")
            doc = reader.readSBMLFromString(sbml_string)
            source_name = None
        elif isinstance(filename_or_string, str) and (
            filename_or_string.lstrip().startswith("<") or "\n" in filename_or_string
        ):
            doc = reader.readSBMLFromString(filename_or_string)
            source_name = None
        else:
            filename = os.path.abspath(filename_or_string)
            doc = reader.readSBMLFromFile(filename)
            source_name = filename

        self._check_sbml_errors(doc, force)
        self._check_sbml_level_version(doc)

        sbml_model = doc.getModel()
        if sbml_model is None:
            raise SbmlImportError("No model found in SBML file")

        model_id = sbml_model.getId()
        if model_id:
            name = model_id
        elif source_name is not None:
            name = os.path.splitext(os.path.basename(source_name))[0]
        else:
            name = "model"
        self.model.name = _sanitize_id(name) or "model"

        self._force = force
        self._log = pysb.logging.get_logger(__name__)
        self._id_map = {}  # SBML ID -> PySB component name
        self._species_obs = {}  # SBML species ID -> Observable
        self._func_defs = {}  # SBML function definition ID -> callable
        self._boundary_species = set()  # SBML species IDs with boundaryCondition=True

        # Collect variable IDs that are governed by rules (not plain Parameters)
        self._assigned_vars = set()  # assignment rules -> Expressions
        self._rate_rule_vars = set()  # rate rules -> Monomers + ODE rules
        for i in range(sbml_model.getNumRules()):
            rule = sbml_model.getRule(i)
            tc = rule.getTypeCode()
            if tc == libsbml.SBML_ASSIGNMENT_RULE:
                self._assigned_vars.add(rule.getVariable())
            elif tc == libsbml.SBML_RATE_RULE:
                var_id = rule.getVariable()
                if var_id in self._rate_rule_vars:
                    self._warn_or_except(
                        'Multiple rate rules for variable "{}"; only the first '
                        "will be used".format(var_id)
                    )
                else:
                    self._rate_rule_vars.add(var_id)
            elif tc == libsbml.SBML_ALGEBRAIC_RULE:
                self._warn_or_except(
                    "Algebraic rules are not supported and cannot be "
                    "represented in PySB (algebraic rule {} skipped)".format(
                        rule.getId() or i
                    )
                )

        if sbml_model.getNumEvents() > 0:
            self._warn_or_except(
                "SBML Events are not supported and will be ignored "
                "({} event(s) found)".format(sbml_model.getNumEvents())
            )

        self._parse_function_definitions(sbml_model)
        self._parse_compartments(sbml_model)
        self._parse_species(sbml_model)
        self._parse_parameters(sbml_model)
        self._parse_initial_assignments(sbml_model)
        self._parse_assignment_rules(sbml_model)
        self._parse_rate_rules(sbml_model)
        self._parse_reactions(sbml_model)

    def _check_sbml_errors(self, doc, force):
        """Raise or warn if the SBML document contains fatal error messages.

        Iterates over libsbml error log entries and collects those with
        severity at or above ``LIBSBML_SEV_ERROR``.  If any are found and
        *force* is False, raises :class:`SbmlImportError`; with *force* True
        a :mod:`warnings` warning is issued and the caller may continue with
        a partial model.
        """
        errors = []
        for i in range(doc.getNumErrors()):
            error = doc.getError(i)
            if error.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
                errors.append("[{}] {}".format(error.getErrorId(), error.getMessage()))
        if errors:
            msg = "SBML file contains errors:\n" + "\n".join(errors)
            if force:
                warnings.warn(msg)
            else:
                raise SbmlImportError(msg)

    def _check_sbml_level_version(self, doc):
        """Warn if the SBML document level/version is outside the validated range.

        The importer has been validated against SBML Level 2 Versions 1–5 and
        Level 3 Versions 1–2.  Level 1 is readable but has limited features and
        has not been thoroughly validated.  Unknown future levels/versions issue
        a warning so the user is aware that results may be incorrect.
        """
        level = doc.getLevel()
        version = doc.getVersion()
        if level == 1:
            warnings.warn(
                "SBML Level 1 has limited support and has not been thoroughly "
                "validated by the PySB importer. Consider upgrading to Level 2 "
                "or 3."
            )
        elif level == 2 and version > 5:
            warnings.warn(
                "SBML Level 2 Version {} is not a known release; the importer "
                "has been validated against Level 2 Versions 1–5.".format(version)
            )
        elif level == 3 and version > 2:
            warnings.warn(
                "SBML Level 3 Version {} has not been validated by the PySB "
                "importer (validated against Level 3 Versions 1–2).".format(version)
            )
        elif level > 3:
            warnings.warn(
                "SBML Level {} is unknown to this importer; import may fail or "
                "produce incorrect results.".format(level)
            )

    def _warn_or_except(self, msg):
        """Issue a warning or raise :class:`SbmlImportError` depending on *force*.

        When ``self._force`` is True the message is emitted via
        :mod:`warnings`; otherwise :class:`SbmlImportError` is raised.
        This gives callers a uniform way to handle unsupported constructs.
        """
        if self._force:
            warnings.warn(msg)
        else:
            raise SbmlImportError(msg)

    def _parse_function_definitions(self, sbml_model):
        """Parse SBML FunctionDefinition elements into Python callables.

        Each function definition is stored in ``self._func_defs`` and later
        added to the local namespace used for SBML expression parsing.
        """
        for i in range(sbml_model.getNumFunctionDefinitions()):
            fd = sbml_model.getFunctionDefinition(i)
            fd_id = fd.getId()
            math = fd.getMath()

            if math is None or math.getType() != libsbml.AST_LAMBDA:
                continue

            # Children are formal arguments followed by the body expression
            n_children = math.getNumChildren()
            if n_children == 0:
                continue
            arg_names = [math.getChild(j).getName() for j in range(n_children - 1)]
            body_node = math.getChild(n_children - 1)
            body_formula = libsbml.formulaToL3String(body_node)
            if body_formula is None:
                continue

            # Build a sympy lambda: parse the body with symbolic args so that
            # calls can be substituted when the function is used in a rate law.
            arg_syms = [sympy.Symbol(a) for a in arg_names]
            try:
                body_expr = parse_bngl_expr(
                    body_formula, local_dict={a: s for a, s in zip(arg_names, arg_syms)}
                )
            except Exception as exc:
                self._warn_or_except(
                    'Could not parse function definition "{}": {}'.format(fd_id, exc)
                )
                continue

            def _make_callable(arg_syms, body_expr):
                def fn(*args):
                    return body_expr.subs(dict(zip(arg_syms, args)))

                return fn

            self._func_defs[fd_id] = _make_callable(arg_syms, body_expr)

    def _parse_compartments(self, sbml_model):
        """Create PySB Compartments and matching size Parameters.

        Each SBML ``<compartment>`` becomes a :class:`~pysb.core.Compartment`
        plus a :class:`~pysb.core.Parameter` named ``<id>_size`` holding the
        initial size.  Compartment hierarchies (the ``outside`` attribute) are
        respected: the parent compartment must appear before the child in the
        SBML document, which is the standard ordering.
        """
        for i in range(sbml_model.getNumCompartments()):
            cpt = sbml_model.getCompartment(i)
            cpt_id = cpt.getId()
            cpt_name = _sanitize_id(cpt_id)
            self._id_map[cpt_id] = cpt_name

            size_val = cpt.getSize() if cpt.isSetSize() else 1.0
            size_param = self.parameter(cpt_name + "_size", size_val, nonnegative=False)

            parent = None
            outside = cpt.getOutside() if cpt.isSetOutside() else None
            if outside:
                parent_name = self._id_map.get(outside, _sanitize_id(outside))
                parent = self.model.compartments.get(parent_name)

            dims = cpt.getSpatialDimensions()
            self.compartment(
                name=cpt_name,
                parent=parent,
                dimension=int(dims) if dims else 3,
                size=size_param,
            )

    def _parse_species(self, sbml_model):
        """Create Monomers, Observables, and initial conditions for SBML species.

        Each ``<species>`` element becomes:

        * A site-less :class:`~pysb.core.Monomer`.
        * An :class:`~pysb.core.Observable` ``obs_<id>`` used in rate
          expressions.
        * An initial-condition :class:`~pysb.core.Parameter` ``<id>_0``
          holding the ``initialAmount`` or ``initialConcentration`` value
          (``initialAmount`` takes precedence when both are present, per the
          SBML specification).

        Species with ``boundaryCondition="true"`` are created as *fixed*
        initials so their concentration is held constant throughout simulation,
        **unless** a rate rule governs them; in that case the rate rule drives
        their dynamics and the initial should not be fixed.
        """
        for i in range(sbml_model.getNumSpecies()):
            sp = sbml_model.getSpecies(i)
            sp_id = sp.getId()
            sp_name = _sanitize_id(sp_id)
            self._id_map[sp_id] = sp_name

            mon = self.monomer(sp_name)

            cpt_id = sp.getCompartment()
            cpt = (
                self.model.compartments.get(
                    self._id_map.get(cpt_id, _sanitize_id(cpt_id))
                )
                if cpt_id
                else None
            )

            mon_pat = MonomerPattern(mon, {}, cpt)
            cp = ComplexPattern([mon_pat], cpt)

            # Observable for use in kinetic law and assignment rule expressions
            obs = self.observable("obs_" + sp_name, cp)
            self._species_obs[sp_id] = obs

            # Initial condition
            if sp.isSetInitialAmount():
                init_val = sp.getInitialAmount()
            elif sp.isSetInitialConcentration():
                init_val = sp.getInitialConcentration()
            else:
                init_val = 0.0

            init_param = self.parameter(sp_name + "_0", init_val, nonnegative=False)
            # Track boundary species (used later to exclude them from reaction
            # patterns and rate-division factors).
            if sp.getBoundaryCondition():
                self._boundary_species.add(sp_id)
            # A boundary species whose amount is governed by a rate rule must
            # NOT be fixed: the rate rule drives its dynamics.  Only fix it
            # when there is no rate rule (i.e. it is truly held constant).
            is_fixed = sp.getBoundaryCondition() and sp_id not in self._rate_rule_vars
            self.initial(cp, init_param, fixed=is_fixed)

    def _parse_parameters(self, sbml_model):
        """Create PySB Parameters for plain SBML parameters.

        Parameters governed by ``<assignmentRule>`` or ``<rateRule>`` elements
        are skipped here; they are handled by :meth:`_parse_assignment_rules`
        and :meth:`_parse_rate_rules` respectively.  Parameters whose value
        attribute is not set (valid in SBML when a value will be supplied
        externally) are also skipped.
        """
        for i in range(sbml_model.getNumParameters()):
            param = sbml_model.getParameter(i)
            param_id = param.getId()
            param_name = _sanitize_id(param_id)
            self._id_map[param_id] = param_name

            # Parameters governed by rules are handled separately
            if param_id in self._assigned_vars or param_id in self._rate_rule_vars:
                continue

            if param.isSetValue():
                self.parameter(param_name, param.getValue(), nonnegative=False)

    def _parse_initial_assignments(self, sbml_model):
        """Handle SBML initialAssignment elements.

        These override any initialAmount/initialConcentration values set on
        species, and are the mechanism used by PySB's SBML exporter.
        """
        for i in range(sbml_model.getNumInitialAssignments()):
            ia = sbml_model.getInitialAssignment(i)
            symbol = ia.getSymbol()
            math = ia.getMath()
            if math is None:
                continue

            # We only update species initials; parameter assignments are rare
            sp_name = self._id_map.get(symbol)
            if sp_name is None or sp_name not in self.model.monomers.keys():
                continue

            expr_val = self._formula_to_sympy(math)

            # Replace the placeholder initial with the correct value
            for ic in self.model.initials:
                if (
                    len(ic.pattern.monomer_patterns) == 1
                    and ic.pattern.monomer_patterns[0].monomer.name == sp_name
                ):
                    if isinstance(expr_val, sympy.Expr) and expr_val.is_number:
                        # Numerically constant expression: update the
                        # existing placeholder Parameter value in place.
                        ic.value.value = float(expr_val)
                    elif isinstance(expr_val, sympy.Expr):
                        # Expression with free symbols: try to evaluate it
                        # numerically by substituting known Parameter values.
                        # BNG/PySB require ic.value to be a Parameter, so
                        # we cannot store a raw sympy Mul/Add there.
                        from pysb import Parameter as _Parameter

                        sub_map = {
                            sym: sym.value
                            for sym in expr_val.free_symbols
                            if isinstance(sym, _Parameter)
                        }
                        numeric_val = expr_val.subs(sub_map)
                        if (
                            isinstance(numeric_val, sympy.Expr)
                            and numeric_val.is_number
                        ):
                            ic.value.value = float(numeric_val)
                        else:
                            # Cannot reduce to a number; create a PySB
                            # Expression and store it as ic.value.
                            # (Rare; only happens when a free symbol is not a
                            # known constant Parameter.)
                            expr_comp = self.expression(sp_name + "_init", expr_val)
                            ic.value = expr_comp
                    break

    def _build_formula_locals(self):
        """Build a local namespace for sympy expression parsing."""
        local_dict = dict(_SBML_MATH_FUNCTIONS)

        # Map species IDs to their observables
        for sp_id, obs in self._species_obs.items():
            local_dict[sp_id] = obs

        # Map SBML IDs and sanitised names to model components.
        # Compartments are not sympy-compatible; map them to their size
        # parameters instead (compartment volume as used in kinetic laws).
        from pysb import Compartment as _Compartment

        for orig_id, pysb_name in self._id_map.items():
            if orig_id in local_dict:
                continue
            try:
                comp = self.model.all_components()[pysb_name]
            except KeyError:
                continue
            if isinstance(comp, _Compartment):
                size_name = pysb_name + "_size"
                try:
                    comp = self.model.parameters[size_name]
                except KeyError:
                    continue
            local_dict[orig_id] = comp
            local_dict[pysb_name] = comp

        # Map SBML csymbol 'time' to PySB's time special symbol so that
        # time-varying kinetic laws are represented correctly in Expressions.
        local_dict["time"] = pysb_time

        # Include user-defined SBML functions
        local_dict.update(self._func_defs)

        return local_dict

    def _formula_to_sympy(self, ast_node):
        """Convert a libsbml ASTNode to a sympy expression.

        Serialises *ast_node* to an L3 formula string via
        ``libsbml.formulaToL3String``, then parses it with
        :func:`~pysb.bng.parse_bngl_expr` using the local namespace built by
        :meth:`_build_math_locals`.  Returns ``sympy.Integer(0)`` when the
        node cannot be serialised or parsed (also calls :meth:`_warn_or_except`
        so the failure is visible to the caller).
        """
        formula = libsbml.formulaToL3String(ast_node)
        if formula is None:
            return sympy.Integer(0)

        try:
            return parse_bngl_expr(formula, local_dict=self._build_formula_locals())
        except Exception as exc:
            self._warn_or_except(
                'Could not parse SBML math "{}": {}'.format(formula, exc)
            )
            return sympy.Integer(0)

    def _parse_assignment_rules(self, sbml_model):
        """Create PySB Expressions for SBML assignmentRule elements.

        Each rule ``variable = f(...)`` becomes an
        :class:`~pysb.core.Expression` whose value is recomputed at every
        time point.  Rules targeting compartment IDs (dynamic compartment
        sizes) are not supported and trigger :meth:`_warn_or_except`.  Rules
        whose variable name already maps to an existing model component (e.g.
        a species observable) are silently skipped.
        """
        cpt_ids = {
            sbml_model.getCompartment(i).getId()
            for i in range(sbml_model.getNumCompartments())
        }

        for i in range(sbml_model.getNumRules()):
            rule = sbml_model.getRule(i)
            if rule.getTypeCode() != libsbml.SBML_ASSIGNMENT_RULE:
                continue

            var_id = rule.getVariable()
            var_name = _sanitize_id(var_id)
            self._id_map[var_id] = var_name

            if var_id in cpt_ids:
                self._warn_or_except(
                    'Assignment rule targeting compartment "{}" (dynamic '
                    "compartment size) is not supported and will be "
                    "ignored".format(var_id)
                )
                continue

            # Skip if already a model component (e.g., species observables)
            if var_name in self.model.all_components().keys():
                continue

            math = rule.getMath()
            if math is None:
                continue

            expr_val = self._formula_to_sympy(math)
            self.expression(var_name, expr_val)

    def _parse_rate_rules(self, sbml_model):
        """Handle SBML rateRule elements.

        Each rate-rule variable X is treated as a species (Monomer with no
        sites) and its ODE is encoded as a synthesis rule::

            None >> X()   with rate = full RHS expression

        With a function-based rate, PySB/BNG use the expression value directly
        as the ODE flux, so this faithfully encodes ``dX/dt = f(...)``.  The
        concentration can go negative (e.g. membrane voltage), which is valid
        for ScipyOdeSimulator.
        """
        # If the model has compartments, rate-rule variables (which are SBML
        # parameters, not species) are assigned to the first compartment so
        # that PySB patterns are concrete.  For truly ODE-only models this is
        # a formal assignment that does not affect the mathematics.
        cpt_list = list(self.model.compartments.values())
        default_cpt = cpt_list[0] if cpt_list else None

        # Collect compartment IDs for dynamic-compartment detection
        cpt_ids = {
            sbml_model.getCompartment(i).getId()
            for i in range(sbml_model.getNumCompartments())
        }

        # Create Monomers/Observables/Initials for rate-rule variables that
        # are not already SBML species
        for var_id in self._rate_rule_vars:
            if var_id in self._species_obs:
                continue

            if var_id in cpt_ids:
                self._warn_or_except(
                    'Rate rule targeting compartment "{}" (dynamic compartment '
                    "size) is not supported and will be ignored".format(var_id)
                )
                continue

            var_name = _sanitize_id(var_id)
            self._id_map[var_id] = var_name

            # Initial value comes from the SBML parameter definition
            param = sbml_model.getParameter(var_id)
            init_val = param.getValue() if (param and param.isSetValue()) else 0.0

            mon = self.monomer(var_name)
            cp = ComplexPattern([MonomerPattern(mon, {}, default_cpt)], default_cpt)
            obs = self.observable("obs_" + var_name, cp)
            self._species_obs[var_id] = obs

            init_p = self.parameter(var_name + "_0", init_val, nonnegative=False)
            self.initial(cp, init_p)

        # Create one production rule per rate rule
        # Track which variables have already been assigned a production rule
        # so that duplicate rate rules (warned about during collection) are
        # not processed a second time.
        processed_rate_vars = set()
        for i in range(sbml_model.getNumRules()):
            rule = sbml_model.getRule(i)
            if rule.getTypeCode() != libsbml.SBML_RATE_RULE:
                continue

            var_id = rule.getVariable()

            # Skip duplicates: only the first rule for each variable is used.
            if var_id not in self._rate_rule_vars:
                continue
            if var_id in processed_rate_vars:
                continue
            processed_rate_vars.add(var_id)

            var_name = self._id_map.get(var_id, _sanitize_id(var_id))

            math = rule.getMath()
            if math is None:
                continue

            try:
                rate_expr = self._formula_to_sympy(math)
            except Exception as exc:
                self._warn_or_except(
                    'Could not parse rate rule for "{}": {}'.format(var_id, exc)
                )
                continue

            rate_comp = self.expression(var_name + "_rate", rate_expr)

            mon = self.model.monomers[var_name]
            cp = ComplexPattern([MonomerPattern(mon, {}, default_cpt)], default_cpt)
            rule_exp = RuleExpression(
                ReactionPattern([]),  # null (no reactants)
                ReactionPattern([cp]),  # >> X()
                is_reversible=False,
            )
            self.rule(var_name + "_ode", rule_exp, rate_comp)

    def _expr_or_param(self, expr, name):
        """Return a bare Parameter if *expr* simplifies to one, else an Expression.

        After dividing the SBML kinetic law by the reactant observable product,
        many simple mass-action rates reduce to a single PySB
        :class:`~pysb.core.Parameter`.  In that case creating a wrapping
        :class:`~pysb.core.Expression` adds noise with no benefit; it is
        cleaner to use the Parameter directly.

        Parameters
        ----------
        expr : sympy.Expr
            The simplified rate sympy expression.
        name : str
            The name to give the Expression if one must be created.

        Returns
        -------
        pysb.core.Parameter or pysb.core.Expression
        """
        from pysb import Parameter as _Parameter

        # A bare Parameter atom has exactly one free symbol and that symbol IS
        # a PySB Parameter object already in the model.
        free = expr.free_symbols
        if len(free) == 1:
            sym = next(iter(free))
            if (
                isinstance(sym, _Parameter)
                and sym in self.model.parameters.values()
                and expr == sym
            ):
                return sym
        return self.expression(name, expr)

    def _compartment_volume(self, cpt_id):
        """Return the sympy size symbol for a compartment ID.

        Looks up the ``<compartmentID>_size`` :class:`~pysb.core.Parameter`
        created by :meth:`_parse_compartments`.  Returns ``sympy.Integer(1)``
        when no compartment is specified, when the parameter does not exist, or
        when the compartment size is trivially 1 (no-op for division).
        """
        if not cpt_id:
            return sympy.Integer(1)
        cpt_name = self._id_map.get(cpt_id, _sanitize_id(cpt_id))
        size_param = self.model.parameters.get(cpt_name + "_size")
        if size_param is None or size_param.value == 1.0:
            return sympy.Integer(1)
        return size_param

    def _reaction_volume(self, sbml_model, rxn):
        """Return the sympy compartment-size symbol for a reaction.

        SBML kinetic laws give flux in **amount per time**.  PySB/BNG build
        concentration ODEs, so each flux must be divided by the volume of the
        reaction compartment.  This method returns the appropriate size
        :class:`~pysb.core.Parameter` (or ``sympy.Integer(1)`` when no
        compartment can be determined or the size is trivially 1).

        The compartment is taken from (in order of preference):

        1. The reaction's own ``compartment`` attribute (SBML Level 3).
        2. The compartment of the first non-boundary reactant species.
        3. The compartment of the first non-boundary product species.
        4. The first compartment defined in the model.
        5. ``sympy.Integer(1)`` (no compartments, bare ODE model).
        """
        # 1. SBML Level 3 reaction compartment attribute
        cpt_id = rxn.getCompartment() if rxn.isSetCompartment() else None

        # 2. First non-boundary reactant
        if not cpt_id:
            for j in range(rxn.getNumReactants()):
                sr = rxn.getReactant(j)
                sp_id = sr.getSpecies()
                if sp_id in self._boundary_species:
                    continue
                sp = sbml_model.getSpecies(sp_id)
                if sp and sp.isSetCompartment():
                    cpt_id = sp.getCompartment()
                    break

        # 3. First non-boundary product
        if not cpt_id:
            for j in range(rxn.getNumProducts()):
                sr = rxn.getProduct(j)
                sp_id = sr.getSpecies()
                if sp_id in self._boundary_species:
                    continue
                sp = sbml_model.getSpecies(sp_id)
                if sp and sp.isSetCompartment():
                    cpt_id = sp.getCompartment()
                    break

        # 4. First model compartment
        if not cpt_id and sbml_model.getNumCompartments() > 0:
            cpt_id = sbml_model.getCompartment(0).getId()

        return self._compartment_volume(cpt_id)

    def _product_volume(self, sbml_model, rxn):
        """Return the sympy compartment-size symbol for the product side.

        Mirrors :meth:`_reaction_volume` but looks at the product compartment
        rather than the reactant compartment.  Used to detect and handle
        cross-compartment transport reactions where the reactant and product
        compartments differ (and therefore require different volume divisors for
        each species ODE).

        The compartment is taken from (in order of preference):

        1. The reaction's own ``compartment`` attribute (SBML Level 3).
        2. The compartment of the first non-boundary product species.
        3. ``sympy.Integer(1)`` (no products / no compartment information).
        """
        # 1. SBML Level 3 reaction compartment attribute
        cpt_id = rxn.getCompartment() if rxn.isSetCompartment() else None

        # 2. First non-boundary product
        if not cpt_id:
            for j in range(rxn.getNumProducts()):
                sr = rxn.getProduct(j)
                sp_id = sr.getSpecies()
                if sp_id in self._boundary_species:
                    continue
                sp = sbml_model.getSpecies(sp_id)
                if sp and sp.isSetCompartment():
                    cpt_id = sp.getCompartment()
                    break

        return self._compartment_volume(cpt_id)

    def _obs_product_for_refs(self, list_of_refs):
        """Build the sympy product of Observable objects for a species-reference list.

        For each species reference in *list_of_refs*, looks up the corresponding
        :class:`~pysb.core.Observable` (created in :meth:`_parse_species`) and
        raises it to the power of its stoichiometry.  Returns
        ``sympy.Integer(1)`` when the list is empty (source reactions).

        This product is used to divide the SBML kinetic law and recover the
        intrinsic rate expression expected by PySB/BNG rules. PySB/BNG multiply
        the rule rate by the reactant species counts, so the kinetic law (which
        already includes those factors) must be divided by them first.

        The Observable *objects* are used directly (not their string names) so
        that sympy can cancel identical Observable atoms in the numerator and
        denominator of the division.

        Parameters
        ----------
        list_of_refs : libsbml.ListOfSpeciesReferences
            Reactant or product species-reference list from a libsbml reaction.

        Returns
        -------
        sympy.Expr
            Product of ``obs ** stoich`` for each species reference.
        """
        product = sympy.Integer(1)
        for j in range(list_of_refs.size()):
            sr = list_of_refs.get(j)
            sp_id = sr.getSpecies()
            # Boundary species are excluded from PySB rule patterns, so their
            # observables must not be included in the division factor either.
            if sp_id in self._boundary_species:
                continue
            obs = self._species_obs.get(sp_id)
            if obs is None:
                continue
            stoich = 1
            if sr.isSetStoichiometry():
                s = sr.getStoichiometry()
                stoich = (
                    int(round(s)) if math.isclose(s, round(s), rel_tol=1e-9) else int(s)
                )
            product = product * obs**stoich
        return product

    def _combinatorial_correction(self, list_of_refs):
        """Compute the combinatorial correction factor for a species-reference list.

        PySB/BNG divides the rule rate by ``n!`` for each species that appears
        with stoichiometry *n* (identical-reactant symmetry correction).  The
        SBML kinetic law encodes the *net flux* directly and does not include
        this factor, so the importer must multiply the intrinsic rate by the
        same ``n!`` to counteract BNG's division.

        For example, a homodimerisation reaction ``A + A -> B`` (stoichiometry
        2 for A) gets a correction factor of ``2! = 2``.  A reaction with
        distinct reactants (all stoichiometries equal to 1) gets a factor of 1
        and no correction is needed.

        Parameters
        ----------
        list_of_refs : libsbml.ListOfSpeciesReferences
            Reactant (or product) species-reference list from a libsbml reaction.

        Returns
        -------
        int
            Product of ``stoich!`` over all unique species in *list_of_refs*.
        """
        # Accumulate stoichiometry per unique species ID
        stoich_by_species = {}
        for j in range(list_of_refs.size()):
            sr = list_of_refs.get(j)
            sp_id = sr.getSpecies()
            # Boundary species are excluded from PySB rule patterns, so BNG
            # applies no symmetry correction for them.  Skip them here.
            if sp_id in self._boundary_species:
                continue
            stoich = 1
            if sr.isSetStoichiometry():
                s = sr.getStoichiometry()
                stoich = (
                    int(round(s)) if math.isclose(s, round(s), rel_tol=1e-9) else int(s)
                )
            stoich_by_species[sp_id] = stoich_by_species.get(sp_id, 0) + stoich
        correction = 1
        for stoich in stoich_by_species.values():
            correction *= math.factorial(stoich)
        return correction

    def _stoich_refs_to_patterns(self, sbml_model, list_of_refs):
        """Convert a ListOfSpeciesReferences to a list of ComplexPatterns.

        Each species reference is expanded into *stoichiometry* copies of a
        single-monomer :class:`~pysb.core.ComplexPattern`.  Non-integer
        stoichiometry values are truncated and a :mod:`warnings` warning is
        issued.  A stoichiometry of zero produces no patterns (effectively
        removing that species reference from the rule).

        Species with ``boundaryCondition="true"`` are **excluded** from the
        returned pattern list.  Per the SBML specification, boundary species
        are external: reactions proceed and can reference them in kinetic
        laws, but their amounts are not changed by reactions (only by rate
        rules or assignment rules).  Including them in the PySB rule pattern
        would cause BNG to subtract/add them in the ODE, which is incorrect.
        """
        patterns = []
        for j in range(list_of_refs.size()):
            sr = list_of_refs.get(j)
            sp_id = sr.getSpecies()
            sp_name = self._id_map.get(sp_id, _sanitize_id(sp_id))

            # Boundary species: skip; their amounts are not affected by rxns.
            sbml_sp = sbml_model.getSpecies(sp_id)
            if sbml_sp is not None and sbml_sp.getBoundaryCondition():
                continue

            stoich = 1
            if sr.isSetStoichiometry():
                s = sr.getStoichiometry()
                if math.isclose(s, round(s), rel_tol=1e-9):
                    stoich = int(round(s))
                else:
                    warnings.warn(
                        'Non-integer stoichiometry ({}) for species "{}" is '
                        "not supported; truncating to {}".format(s, sp_id, int(s))
                    )
                    stoich = int(s)

            mon = self.model.monomers[sp_name]
            sp = sbml_model.getSpecies(sp_id)
            cpt_id = sp.getCompartment() if sp else None
            cpt = (
                self.model.compartments.get(
                    self._id_map.get(cpt_id, _sanitize_id(cpt_id))
                )
                if cpt_id
                else None
            )

            for _ in range(stoich):
                patterns.append(ComplexPattern([MonomerPattern(mon, {}, cpt)], cpt))

        return patterns

    def _parse_reactions(self, sbml_model):
        """Create PySB Rules for SBML reaction elements.

        Each ``<reaction>`` normally becomes one or two non-reversible
        :class:`~pysb.core.Rule` objects.  Cross-compartment transport
        reactions (reactant compartment ≠ product compartment) are split into
        an additional pair of rules; see below.

        **ODE correctness**: PySB/BNG rules with :class:`~pysb.core.Expression`
        rates multiply the expression value by the reactant species counts when
        building ODEs (mass-action combinatorics).  SBML kinetic laws already
        include the reactant species as factors (they express the net *flux*
        directly).  The importer therefore divides the kinetic law by the
        product of reactant observable symbols (each raised to its
        stoichiometry) to recover the intrinsic rate expression:

        .. math::

            \\text{pysb\\_rate} = \\frac{J}{\\prod_i [S_i]^{s_i}}

        For source reactions (no reactants) the kinetic law is used as-is.

        **Compartment volume**: SBML kinetic laws give flux in amount/time.
        PySB builds concentration-based ODEs, so the flux is divided by the
        reaction compartment volume (``dC/dt = J/V``).

        **Cross-compartment transport**: when the reactant compartment
        ``V_src`` and product compartment ``V_dst`` differ, a single rule
        would divide by the same volume on both sides, producing the wrong ODE
        for one side.  Instead, the reaction is split into:

        * A **degradation rule** ``reactants → ∅`` with rate
          ``J / (V_src · ∏[S_i]^{s_i})``, giving ``dR/dt = −J/V_src``.
        * A **synthesis rule** ``∅ → products`` with rate ``J / V_dst²``.
          PySB/BNG internally multiply synthesis rates by the product
          compartment volume, so the effective ODE contribution is
          ``V_dst · (J/V_dst²) = J/V_dst``, giving ``dP/dt = +J/V_dst``.

        **Reversible reactions**: when ``reversible="true"``, the kinetic law
        encodes the net flux ``J = J_\\text{fwd} - J_\\text{rev}``.
        :func:`_split_reversible_rate` is used to separate positive and
        negative additive terms into the forward and reverse fluxes.  Each
        half-flux is then divided by the product of the respective species
        (reactants for the forward half, products for the reverse half) and
        encoded as a separate non-reversible :class:`~pysb.core.Rule`.  If the
        split fails (non-separable kinetic law), the reaction is encoded as a
        single forward rule and a warning is issued.

        The ``fast`` attribute (quasi-steady-state flag, deprecated in SBML
        Level 3 Version 2) triggers a :mod:`warnings` warning and is otherwise
        ignored.

        Local parameters in kinetic laws (which shadow global parameters)
        are not added to the PySB model; a :mod:`warnings` warning is issued
        listing the affected parameter IDs.

        Reactions without an ``id`` attribute receive the fallback name
        ``r{i}``.
        """
        for i in range(sbml_model.getNumReactions()):
            rxn = sbml_model.getReaction(i)
            rxn_id = rxn.getId()
            rxn_name = _sanitize_id(rxn_id) if rxn_id else "r{}".format(i)

            # The SBML 'fast' attribute signals quasi-steady-state (QSS)
            # treatment.  PySB has no QSS mechanism, so warn if set.
            if rxn.isSetFast() and rxn.getFast():
                warnings.warn(
                    'Reaction "{}" has fast="true" (quasi-steady-state); '
                    "this attribute has no effect in PySB and is "
                    "ignored.".format(rxn_id)
                )

            reactant_pats = self._stoich_refs_to_patterns(
                sbml_model, rxn.getListOfReactants()
            )
            product_pats = self._stoich_refs_to_patterns(
                sbml_model, rxn.getListOfProducts()
            )

            kl = rxn.getKineticLaw()
            if kl is None:
                self._warn_or_except("Reaction {} has no kinetic law".format(rxn_id))
                continue

            # Handle local kinetic-law parameters.
            # In SBML Level 2, parameters are declared inside <kineticLaw>
            # elements via <listOfParameters>.  In Level 3, they appear in
            # <listOfLocalParameters>.  libsbml's getNumParameters() covers
            # both cases; getNumLocalParameters() returns 0 for L2.
            #
            # Local parameters that shadow a global parameter of the same name
            # cannot be added (there is already a PySB Parameter with that
            # name); we warn about the shadowing and use the global.
            # Local parameters for which no global exists are promoted to
            # global PySB Parameters so that the kinetic-law expression can
            # reference them.
            n_local = kl.getNumParameters()
            if n_local > 0:
                shadowed_ids = []
                for j in range(n_local):
                    lp = kl.getParameter(j)
                    lp_id = lp.getId()
                    lp_name = _sanitize_id(lp_id)
                    if lp_name in self.model.parameters.keys():
                        # Global parameter already exists; warn about shadow.
                        shadowed_ids.append(lp_id)
                    elif lp.isSetValue():
                        # No global exists: promote to global Parameter.
                        self.parameter(lp_name, lp.getValue(), nonnegative=False)
                        self._id_map[lp_id] = lp_name
                if shadowed_ids:
                    warnings.warn(
                        'Reaction "{}" has local kinetic-law parameters ({}) '
                        "that shadow global parameters of the same name; the "
                        "global parameters are used.".format(
                            rxn_id, ", ".join(shadowed_ids)
                        )
                    )

            net_flux = self._formula_to_sympy(kl.getMath())

            # SBML kinetic laws give flux in amount/time.  PySB/BNG build
            # concentration-based ODEs, so the flux must be divided by the
            # reaction compartment volume (dC/dt = J/V).
            rxn_vol = self._reaction_volume(sbml_model, rxn)

            is_reversible = rxn.getReversible()

            if is_reversible:
                # Split net flux J = J_fwd - J_rev into forward and reverse
                # half-fluxes, then divide each by the appropriate species
                # product and by the compartment volume to obtain intrinsic
                # rate expressions for PySB/BNG rules.
                fwd_flux, rev_flux = _split_reversible_rate(net_flux)
                if fwd_flux is None or rev_flux is None:
                    warnings.warn(
                        'Reversible reaction "{}" has a kinetic law that cannot '
                        "be split into separable forward and reverse fluxes. "
                        "The reaction is encoded as a single forward rule; "
                        "the reverse direction will be absent from the model.".format(
                            rxn_id
                        )
                    )
                    fwd_flux = net_flux
                    rev_flux = None

                # Forward rule: reactants -> products
                reactant_factor = self._obs_product_for_refs(rxn.getListOfReactants())
                fwd_correction = self._combinatorial_correction(
                    rxn.getListOfReactants()
                )
                fwd_rate_expr = sympy.simplify(
                    fwd_correction * fwd_flux / (rxn_vol * reactant_factor)
                )
                fwd_rate = self._expr_or_param(fwd_rate_expr, rxn_name + "_fwd_rate")

                prod_vol = self._product_volume(sbml_model, rxn)
                if prod_vol != rxn_vol and product_pats:
                    # Cross-compartment: split into deg + synth so each side
                    # is divided by its own compartment volume.
                    self.rule(
                        rxn_name + "_fwd_deg",
                        RuleExpression(
                            ReactionPattern(reactant_pats),
                            ReactionPattern([]),
                            is_reversible=False,
                        ),
                        fwd_rate,
                    )
                    fwd_prod_rate = self._expr_or_param(
                        sympy.simplify(fwd_flux / prod_vol**2),
                        rxn_name + "_fwd_prod_rate",
                    )
                    self.rule(
                        rxn_name + "_fwd_prod",
                        RuleExpression(
                            ReactionPattern([]),
                            ReactionPattern(product_pats),
                            is_reversible=False,
                        ),
                        fwd_prod_rate,
                    )
                else:
                    fwd_rule_exp = RuleExpression(
                        ReactionPattern(reactant_pats),
                        ReactionPattern(product_pats),
                        is_reversible=False,
                    )
                    self.rule(rxn_name + "_fwd", fwd_rule_exp, fwd_rate)

                # Reverse rule: products -> reactants (if split succeeded)
                if rev_flux is not None:
                    product_factor = self._obs_product_for_refs(rxn.getListOfProducts())
                    rev_correction = self._combinatorial_correction(
                        rxn.getListOfProducts()
                    )
                    rev_rate_expr = sympy.simplify(
                        rev_correction * rev_flux / (prod_vol * product_factor)
                    )
                    rev_rate = self._expr_or_param(
                        rev_rate_expr, rxn_name + "_rev_rate"
                    )

                    if prod_vol != rxn_vol and reactant_pats:
                        # Cross-compartment reverse: products -> None, None -> reactants
                        self.rule(
                            rxn_name + "_rev_deg",
                            RuleExpression(
                                ReactionPattern(product_pats),
                                ReactionPattern([]),
                                is_reversible=False,
                            ),
                            rev_rate,
                        )
                        rev_prod_rate = self._expr_or_param(
                            sympy.simplify(rev_flux / rxn_vol**2),
                            rxn_name + "_rev_prod_rate",
                        )
                        self.rule(
                            rxn_name + "_rev_prod",
                            RuleExpression(
                                ReactionPattern([]),
                                ReactionPattern(reactant_pats),
                                is_reversible=False,
                            ),
                            rev_prod_rate,
                        )
                    else:
                        rev_rule_exp = RuleExpression(
                            ReactionPattern(product_pats),
                            ReactionPattern(reactant_pats),
                            is_reversible=False,
                        )
                        self.rule(rxn_name + "_rev", rev_rule_exp, rev_rate)
            else:
                # Irreversible: divide kinetic law by reactant observable
                # product and compartment volume to get the intrinsic rate for
                # BNG rules, then multiply by the combinatorial correction so
                # that PySB/BNG's internal symmetry division cancels out.
                reactant_factor = self._obs_product_for_refs(rxn.getListOfReactants())
                correction = self._combinatorial_correction(rxn.getListOfReactants())
                rate_expr = sympy.simplify(
                    correction * net_flux / (rxn_vol * reactant_factor)
                )
                rate_comp = self._expr_or_param(rate_expr, rxn_name + "_rate")

                prod_vol = self._product_volume(sbml_model, rxn)
                if prod_vol != rxn_vol and reactant_pats and product_pats:
                    # Cross-compartment transport: reactant and product
                    # compartments differ, so a single rule would apply the
                    # same volume divisor to both sides, which is incorrect when
                    # V_src ≠ V_dst.  Split into a degradation rule (correct
                    # V_src for the reactant ODE) and a synthesis rule whose
                    # rate is scaled so that PySB/BNG's implicit V_dst
                    # multiplication yield the correct product ODE
                    # (flux / V_dst).
                    self.rule(
                        rxn_name + "_deg",
                        RuleExpression(
                            ReactionPattern(reactant_pats),
                            ReactionPattern([]),
                            is_reversible=False,
                        ),
                        rate_comp,
                    )
                    prod_rate = self._expr_or_param(
                        sympy.simplify(net_flux / prod_vol**2),
                        rxn_name + "_prod_rate",
                    )
                    self.rule(
                        rxn_name + "_prod",
                        RuleExpression(
                            ReactionPattern([]),
                            ReactionPattern(product_pats),
                            is_reversible=False,
                        ),
                        prod_rate,
                    )
                else:
                    rule_exp = RuleExpression(
                        ReactionPattern(reactant_pats),
                        ReactionPattern(product_pats),
                        is_reversible=False,
                    )
                    self.rule(rxn_name, rule_exp, rate_comp)


def _split_reversible_rate(expr):
    """Split a net-flux sympy expression into forward and reverse parts.

    SBML reversible reactions carry a single kinetic law representing the net
    flux ``J = J_forward - J_reverse``.  PySB reversible rules require two
    *separate* non-negative rate expressions.  This helper attempts the split
    by partitioning the additive terms of *expr* according to the sign of each
    term's leading rational coefficient:

    * Terms whose leading coefficient is **positive** are collected into the
      forward part.
    * Terms whose leading coefficient is **negative** are negated and collected
      into the reverse part.

    Parameters
    ----------
    expr : sympy.Expr
        The net-flux expression to split.

    Returns
    -------
    tuple[sympy.Expr, sympy.Expr] or tuple[None, None]
        ``(forward_expr, reverse_expr)`` if the split is unambiguous, i.e.
        at least one term is positive **and** at least one is negative.
        ``(None, None)`` otherwise (all terms share the same sign, or the
        expression is not a plain sum of terms).
    """
    terms = sympy.Add.make_args(sympy.expand(expr))  # expand then split addends

    forward_terms = []
    reverse_terms = []

    for term in terms:
        coeff, _ = term.as_coeff_Mul()
        if coeff > 0:
            forward_terms.append(term)
        elif coeff < 0:
            reverse_terms.append(-term)  # negate so the rate is positive
        else:
            # Zero coefficient or symbolic coefficient: cannot determine sign
            return None, None

    if not forward_terms or not reverse_terms:
        # All terms have the same sign: not a net-flux expression
        return None, None

    return sympy.Add(*forward_terms), sympy.Add(*reverse_terms)


def _model_from_sbml_libsbml(filename_or_string, force=False):
    """
    Create a PySB Model from an SBML file or string using libsbml.

    Thin wrapper around :class:`SbmlImporter`.  See that class and the
    module docstring for full details of the SBML-to-PySB mapping and
    known limitations.

    Parameters
    ----------
    filename_or_string : str or bytes
        Path to an SBML file **or** a raw SBML XML string/bytes.  A value is
        treated as a string/bytes if it starts with ``<`` or contains a
        newline; otherwise it is interpreted as a file path.
    force : bool, optional
        If False (default), raise :class:`SbmlImportError` on unsupported
        constructs; if True, issue warnings and return a partial model.

    Returns
    -------
    pysb.Model

    """
    importer = SbmlImporter(filename_or_string, force=force)
    return importer.model


def sbml_translator(
    input_file,
    output_file=None,
    convention_file=None,
    naming_conventions=None,
    user_structures=None,
    molecule_id=False,
    atomize=False,
    pathway_commons=False,
    verbose=False,
):
    """
    Run the BioNetGen sbmlTranslator binary to convert SBML to BNGL

    This function runs the external program sbmlTranslator, included with
    BioNetGen, which converts SBML files to BioNetGen language (BNGL). If
    PySB was installed using "conda", you can install sbmlTranslator using
    "conda install -c alubbock atomizer". sbmlTranslator is bundled with
    BioNetGen if BNG is installed by manual download and unzip.

    Generally, PySB users don't need to run this function directly; an SBML
    model can be imported to PySB in a single step with
    :func:`model_from_sbml`. However, users may wish to note the parameters
    for this function, which alter the way the SBML file is processed. These
    parameters can be supplied as ``**kwargs`` to :func:`model_from_sbml`
    when ``use_libsbml=False``.

    For more detailed descriptions of the arguments, see the `sbmlTranslator
    documentation <http://bionetgen.org/index.php/SBML2BNGL>`_.

    Parameters
    ----------
    input_file : string
        SBML input filename
    output_file : string, optional
        BNGL output filename
    convention_file : string, optional
        Conventions filename
    naming_conventions : string, optional
        Naming conventions filename
    user_structures : string, optional
        User structures filename
    molecule_id : bool, optional
        Use SBML molecule IDs (True) or names (False).
        IDs are less descriptive but more BNGL friendly. Use only if the
        generated BNGL has syntactic errors
    atomize : bool, optional
        Atomize the model, i.e. attempt to infer molecular structure and
        build rules from the model (True) or just perform a flat import (False)
    pathway_commons : bool, optional
        Use pathway commons to infer molecule binding. This
        setting requires an internet connection and will query the pathway
        commons web service.
    verbose : bool or int, optional (default: False)
        Sets the verbosity level of the logger. See the logging levels and
        constants from Python's logging module for interpretation of integer
        values. False leaves the logging verbosity unchanged, True is equal
        to DEBUG.

    Returns
    -------
    string
        BNGL output filename
    """
    logger = get_logger(__name__, log_level=verbose)
    sbmltrans_bin = pf.get_path("atomizer")

    sbmltrans_args = [sbmltrans_bin, "-i", input_file]
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".bngl"
    sbmltrans_args.extend(["-o", output_file])

    if convention_file:
        sbmltrans_args.extend(["-c", convention_file])

    if naming_conventions:
        sbmltrans_args.extend(["-n", naming_conventions])

    if user_structures:
        sbmltrans_args.extend(["-u", user_structures])

    if molecule_id:
        sbmltrans_args.append("-id")

    if atomize:
        sbmltrans_args.append("-a")

    if pathway_commons:
        sbmltrans_args.append("-p")

    logger.debug("sbmlTranslator command: " + " ".join(sbmltrans_args))

    p = subprocess.Popen(
        sbmltrans_args, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    if logger.getEffectiveLevel() <= EXTENDED_DEBUG:
        output = "\n".join([line for line in iter(p.stdout.readline, b"")])
        if output:
            logger.log(EXTENDED_DEBUG, "sbmlTranslator output:\n\n" + output)
    (p_out, p_err) = p.communicate()
    if p.returncode:
        raise SbmlTranslationError(p_out.decode("utf-8") + "\n" + p_err.decode("utf-8"))

    return output_file


def model_from_sbml(filename, force=False, cleanup=True, use_libsbml=True, **kwargs):
    """
    Create a PySB Model from a Systems Biology Markup Language (SBML) file.

    By default this uses libsbml for a direct flat import (``use_libsbml=True``).
    To use the legacy BioNetGen sbmlTranslator/Atomizer backend, which can
    attempt to infer higher-level rule-based structure, pass
    ``use_libsbml=False``.

    Parameters
    ----------
    filename : str
        Path to the SBML file.
    force : bool, optional
        If False (default), raise on import errors; if True, warn and continue.
    cleanup : bool, optional
        Delete temporary files on completion (only relevant when
        ``use_libsbml=False``).
    use_libsbml : bool, optional
        If True (default), use the libsbml-based importer.
        If False, use the legacy sbmlTranslator/Atomizer pipeline.
    **kwargs
        Additional keyword arguments forwarded to :func:`sbml_translator`
        when ``use_libsbml=False`` (e.g. ``atomize=True``).

    Returns
    -------
    pysb.Model

    Notes
    -----
    The libsbml importer performs a flat import: each SBML species becomes a
    PySB Monomer with no sites, each reaction becomes a rule with a
    function-based rate Expression, and rate-rule (ODE-only) models are
    also supported.  See the module docstring and :class:`SbmlImporter` for
    the full SBML feature mapping and known limitations.

    For structure inference (atomisation) use ``use_libsbml=False`` and
    pass ``atomize=True``.

    The legacy sbmlTranslator backend requires the sbmlTranslator program
    (also known as Atomizer). If PySB was installed with conda, install it
    via ``conda install -c alubbock atomizer``. It is bundled with BioNetGen
    if BNG is installed by manual download and unzip.

    Examples
    --------
    Import a flat SBML model using the default libsbml backend:

    >>> from pysb.importers.sbml import model_from_sbml
    >>> model = model_from_sbml('my_model.xml')              # doctest: +SKIP

    Use the legacy Atomizer backend to infer rule-based structure:

    >>> model = model_from_sbml('my_model.xml',              # doctest: +SKIP
    ...                         use_libsbml=False,
    ...                         atomize=True)
    """
    if use_libsbml:
        return _model_from_sbml_libsbml(filename, force=force)

    logger = get_logger(__name__, log_level=kwargs.get("verbose"))
    tmpdir = tempfile.mkdtemp()
    logger.debug(
        "Performing SBML to BNGL translation in temporary directory %s" % tmpdir
    )
    try:
        bngl_file = os.path.join(tmpdir, "model.bngl")
        sbml_translator(filename, bngl_file, **kwargs)
        return model_from_bngl(bngl_file, force=force, cleanup=cleanup)
    finally:
        if cleanup:
            shutil.rmtree(tmpdir)


def model_from_biomodels(
    accession_no, force=False, cleanup=True, mirror="ebi", use_libsbml=True, **kwargs
):
    """
    Create a PySB Model based on a BioModels SBML model.

    Downloads the SBML file from BioModels (https://www.ebi.ac.uk/biomodels/)
    and passes it to :func:`model_from_sbml`.

    Parameters
    ----------
    accession_no : str
        A BioModels accession number such as ``'BIOMD0000000001'``. For
        brevity, just the numeric part is also accepted (e.g. ``'1'``).
    force : bool, optional
        If False (default), raise on import errors; if True, warn and continue.
    cleanup : bool, optional
        Delete the downloaded SBML file after import.
    mirror : str, optional
        Which BioModels mirror to use: ``'ebi'`` (default) or ``'caltech'``.
    use_libsbml : bool, optional
        If True (default), use the libsbml-based importer.
        If False, use the legacy sbmlTranslator/Atomizer pipeline.
    **kwargs
        Additional keyword arguments forwarded to :func:`model_from_sbml`.

    Returns
    -------
    pysb.Model

    Examples
    --------
    >>> from pysb.importers.sbml import model_from_biomodels
    >>> model = model_from_biomodels('1')           #doctest: +SKIP
    >>> print(model)                                #doctest: +SKIP
    <Model 'pysb' (monomers: 12, rules: 17, parameters: 37, expressions: 0, ...
    """
    logger = get_logger(__name__, log_level=kwargs.get("verbose"))
    if not BIOMODELS_REGEX.match(accession_no):
        try:
            accession_no = "BIOMD{:010d}".format(int(accession_no))
        except ValueError:
            raise ValueError(
                "accession_no must be an integer or a BioModels "
                "accession number (BIOMDxxxxxxxxxx)"
            )
    logger.info("Importing model {} to PySB".format(accession_no))
    filename = _download_biomodels(accession_no, mirror=mirror)
    try:
        return model_from_sbml(
            filename, force=force, cleanup=cleanup, use_libsbml=use_libsbml, **kwargs
        )
    finally:
        try:
            os.remove(filename)
        except OSError:
            pass


def _download_biomodels(accession_no, mirror):
    try:
        url_fmt = BIOMODELS_URLS[mirror]
    except KeyError:
        raise ValueError(
            'Unknown Biomodels mirror: "{}". Choices are: {}'.format(
                mirror, BIOMODELS_URLS.keys()
            )
        )
    filename, _ = urlretrieve(url_fmt.format(accession_no))
    return filename
