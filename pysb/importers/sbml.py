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

Amounts, concentrations and compartment volumes
-----------------------------------------------

PySB/BioNetGen species values are **amounts** (molecule counts or moles), and
BioNetGen's cBNGL network generator scales rule rates by compartment size
(``1/V`` for each reactant beyond the first, ``V`` for synthesis).  The
importer therefore works with amounts throughout:

* ``initialAmount`` is used as-is; ``initialConcentration`` is multiplied by
  the compartment size.
* In SBML math, a species symbol becomes the species observable (an amount)
  when the species has ``hasOnlySubstanceUnits="true"`` (or lives in a
  zero-dimensional compartment) and ``observable / compartment size`` (a
  concentration) otherwise.
* SBML kinetic laws are fluxes in amount per time, which is exactly what a
  PySB rule contributes to the ODEs once BioNetGen's own volume and
  symmetry scaling are compensated for.  The importer applies that
  compensation, so the simulated amounts match any SBML simulator.

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
       **divided by the product of reactant observables** (PySB/BNG multiply
       the rate by the reactant amounts again) and corrected for BNG's
       symmetry factor and compartment-volume scaling.  Reactions that
       cannot be written this way are encoded as signed synthesis rules, see
       below.
   * - ``<reaction>`` (reversible, ``reversible="true"``) with ``<kineticLaw>``
     - Two non-reversible :class:`~pysb.core.Rule` objects (``<id>_fwd``
       and ``<id>_rev``).  The net-flux kinetic law is split into positive
       and negative additive terms, each encoded as above.  If the law
       cannot be split, the reaction is encoded as signed synthesis rules.
   * - ``<parameter>``
     - :class:`~pysb.core.Parameter` (sign unrestricted, so negative values
       such as reversal potentials are accepted)
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
     - Evaluated at import time and stored as the numeric value of the
       species initial-condition, parameter or compartment-size parameter
   * - ``<functionDefinition>``
     - Expanded inline wherever the function is called
   * - ``csymbol`` ``time``
     - Maps to :data:`pysb.core.time`, PySB's simulation-time symbol
   * - ``boundaryCondition="true"``
     - :class:`~pysb.core.Initial` with ``fixed=True``; the species is
       **excluded from all rule patterns** so BNG does not include it in
       ODEs, but it may still appear in kinetic-law expressions
   * - Integer stoichiometry *n* > 1
     - *n* copies of the reactant/product :class:`~pysb.core.ComplexPattern`
       in the rule, plus the symmetry correction described below

Symmetry correction
-------------------

BioNetGen divides the rate of a rule with identical reactants by the
number of symmetry-equivalent mappings, ``n_kept! * n_consumed!``, where for
each reactant species ``n_kept`` is the number of copies that reappear on the
product side and ``n_consumed`` the number that do not.  The importer
multiplies the rule rate by the same factor so that the ODE flux equals the
SBML kinetic law.

Signed synthesis rules
----------------------

A rule contributes ``rate * prod(reactants)`` to the ODEs, so the kinetic law
has to be divided by the reactant observables.  When that quotient is
undefined at zero reactant amounts (e.g. a net-flux law for an irreversible
reaction, ``(10 * S1 - 2 * S2) / (1 + S1 + S2)``), when a reversible law
cannot be split into forward and reverse terms, when a reaction has
non-integer stoichiometry, or when it spans more than one compartment, the
reaction is encoded instead as one synthesis rule
per changed species, ``None >> S()``, with rate ``stoichiometry * flux``
(negative for consumed species).  This is exact for any kinetic law.

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

* **Non-integer stoichiometry** cannot be written as rule patterns.  Such
  reactions are encoded as signed synthesis rules, which keep the exact
  stoichiometry.

* **Rate-rule variables placed in first compartment.** When a model has
  compartments, rate-rule variables that are SBML parameters (not species)
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

* **Compartment hierarchy**: the ``outside`` attribute is only mapped to a
  PySB parent where the dimensions differ (e.g. a 2D membrane inside a 3D
  volume), as BioNetGen rejects nesting of compartments of equal dimension.

* **Initial assignments** are evaluated once at import time; a parameter set
  by an initial assignment does not track later changes to its inputs.

* **Local kinetic-law parameters** become model Parameters.  They keep their
  SBML id when it is unique in the model; otherwise they are named
  ``<reaction id>_<parameter id>``.

* **Unnamed reactions**: reactions with no ``id`` attribute are assigned the
  fallback name ``r0``, ``r1``, …, ``rN``.

* **Multiple rate rules on the same variable** are not supported; the second
  rule encountered for a given variable raises :class:`SbmlImportError` (or
  warns with ``force=True``) because duplicate PySB rules are not valid.

* **MathML**: ``delay``, ``rateOf`` and reaction ids used as symbols are not
  supported and raise :class:`SbmlImportError` (or warn with ``force=True``).
"""

from pysb.importers.bngl import model_from_bngl
from pysb.builder import Builder
from pysb.core import (
    MonomerPattern,
    ComplexPattern,
    RuleExpression,
    ReactionPattern,
    Expression,
    Observable,
    Parameter,
    time as pysb_time,
)
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
from collections import Counter
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


class _UnsupportedMath(Exception):
    """Raised internally when SBML math cannot be converted to sympy."""


def _sanitize_id(sbml_id):
    """Convert an SBML ID to a valid Python/PySB identifier"""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", sbml_id)
    if name and name[0].isdigit():
        name = "s_" + name
    return name or "unnamed"


def _ast_names(node):
    """Return the set of names (symbols and function calls) used in an AST"""
    names = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n.isName() or n.getType() == libsbml.AST_FUNCTION:
            names.add(n.getName())
        stack.extend(n.getChild(j) for j in range(n.getNumChildren()))
    return names


def _copies(stoichiometry):
    """Number of pattern copies for a (possibly non-integer) stoichiometry"""
    if math.isclose(stoichiometry, round(stoichiometry), rel_tol=1e-9):
        return int(round(stoichiometry))
    return int(stoichiometry)


def _is_integral(exact, copies):
    """True if exact stoichiometries (by species) equal the pattern copies"""
    return all(
        math.isclose(exact[sp], copies.get(sp, 0), rel_tol=1e-9, abs_tol=1e-12)
        for sp in exact
    )


def _simple_functions():
    """Map libsbml AST function types to one-argument sympy callables"""
    one = sympy.Integer(1)
    return {
        libsbml.AST_FUNCTION_ABS: sympy.Abs,
        libsbml.AST_FUNCTION_CEILING: sympy.ceiling,
        libsbml.AST_FUNCTION_FLOOR: sympy.floor,
        libsbml.AST_FUNCTION_EXP: sympy.exp,
        libsbml.AST_FUNCTION_LN: sympy.log,
        libsbml.AST_FUNCTION_FACTORIAL: sympy.factorial,
        libsbml.AST_FUNCTION_SIN: sympy.sin,
        libsbml.AST_FUNCTION_COS: sympy.cos,
        libsbml.AST_FUNCTION_TAN: sympy.tan,
        libsbml.AST_FUNCTION_SEC: lambda x: one / sympy.cos(x),
        libsbml.AST_FUNCTION_CSC: lambda x: one / sympy.sin(x),
        libsbml.AST_FUNCTION_COT: lambda x: one / sympy.tan(x),
        libsbml.AST_FUNCTION_SINH: sympy.sinh,
        libsbml.AST_FUNCTION_COSH: sympy.cosh,
        libsbml.AST_FUNCTION_TANH: sympy.tanh,
        libsbml.AST_FUNCTION_SECH: lambda x: one / sympy.cosh(x),
        libsbml.AST_FUNCTION_CSCH: lambda x: one / sympy.sinh(x),
        libsbml.AST_FUNCTION_COTH: lambda x: one / sympy.tanh(x),
        libsbml.AST_FUNCTION_ARCSIN: sympy.asin,
        libsbml.AST_FUNCTION_ARCCOS: sympy.acos,
        libsbml.AST_FUNCTION_ARCTAN: sympy.atan,
        libsbml.AST_FUNCTION_ARCSEC: lambda x: sympy.acos(one / x),
        libsbml.AST_FUNCTION_ARCCSC: lambda x: sympy.asin(one / x),
        libsbml.AST_FUNCTION_ARCCOT: lambda x: sympy.atan(one / x),
        libsbml.AST_FUNCTION_ARCSINH: sympy.asinh,
        libsbml.AST_FUNCTION_ARCCOSH: sympy.acosh,
        libsbml.AST_FUNCTION_ARCTANH: sympy.atanh,
        libsbml.AST_FUNCTION_ARCSECH: lambda x: sympy.acosh(one / x),
        libsbml.AST_FUNCTION_ARCCSCH: lambda x: sympy.asinh(one / x),
        libsbml.AST_FUNCTION_ARCCOTH: lambda x: sympy.atanh(one / x),
    }


def _relational_functions():
    return {
        libsbml.AST_RELATIONAL_EQ: sympy.Eq,
        libsbml.AST_RELATIONAL_NEQ: sympy.Ne,
        libsbml.AST_RELATIONAL_LT: sympy.Lt,
        libsbml.AST_RELATIONAL_LEQ: sympy.Le,
        libsbml.AST_RELATIONAL_GT: sympy.Gt,
        libsbml.AST_RELATIONAL_GEQ: sympy.Ge,
    }


def _unsafe_quotient(expr, zero_symbols):
    """True if *expr* may be undefined when any of *zero_symbols* is zero.

    Factors of the symbols that were divided out have already cancelled
    automatically, so what remains is singular at zero only if the quotient
    could not be simplified (or never was well defined).  Substituting zero
    detects this without the cost of a full symbolic cancellation, which can
    be prohibitive for large kinetic laws.
    """
    for sym in zero_symbols:
        at_zero = expr.subs(sym, 0)
        if at_zero.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo):
            return True
    return False


class SbmlImporter(Builder):
    """Import a flat SBML model using libsbml.

    Reads an SBML file and constructs a PySB model.  The importer is a
    :class:`~pysb.builder.Builder` subclass, so the resulting model is
    available as ``importer.model`` after construction.

    Parameters
    ----------
    filename_or_string : str or bytes
        Path to the SBML file (any SBML Level/Version supported by libsbml),
        or the SBML document itself as a string or bytes.
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
    are prefixed with ``s_``.  Names generated by the importer (``obs_<id>``,
    ``<id>_0``, ``<id>_size``, ``<id>_rate``, ...) are made unique with
    respect to every SBML id by appending underscores.

    **Species/observables**: each SBML ``<species>`` element becomes a
    :class:`~pysb.core.Monomer` named after its SBML ID (sanitised) and a
    corresponding :class:`~pysb.core.Observable` named ``obs_<id>``, whose
    value is the species amount.  Initial conditions are created as
    ``<id>_0`` :class:`~pysb.core.Parameter` objects holding amounts.

    **Reactions**: each SBML ``<reaction>`` becomes one or two
    :class:`~pysb.core.Rule` objects (or several signed synthesis rules, see
    the module documentation) whose function-based rate reproduces the
    kinetic law as an amount flux.

    **Parameters**: SBML ``<parameter>`` elements become PySB
    :class:`~pysb.core.Parameter` objects of unrestricted sign, so that
    negative values (e.g. reversal potentials in electrophysiology models)
    are accepted without error.

    **Compartments**: each ``<compartment>`` becomes both a PySB
    :class:`~pysb.core.Compartment` and a size
    :class:`~pysb.core.Parameter` named ``<id>_size``.  The size parameter
    is used wherever the compartment ID appears in SBML math, and to convert
    species concentrations to amounts.

    **Assignment rules**: ``<assignmentRule>`` elements create PySB
    :class:`~pysb.core.Expression` objects whose value tracks the rule RHS at
    every time point.  The corresponding SBML parameter is *not* added to the
    model as a plain ``Parameter``.  Rules may refer to each other in any
    order.

    **Rate rules**: ``<rateRule>`` elements (ODE-only models) are
    represented as synthesis rules ``None >> X()`` with a function-based rate
    equal to the full RHS.

    **FunctionDefinitions**: SBML ``<functionDefinition>`` elements are
    expanded inline wherever they are called.

    **Initial assignments**: ``<initialAssignment>`` elements override the
    initial value of species, parameters and compartment sizes.  They are
    evaluated numerically at import time.  This is the mechanism used by
    PySB's own SBML exporter, so round-trip import/export is supported.

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
        self._sym = {}  # SBML ID -> sympy object used in SBML math
        self._species_obs = {}  # SBML species ID -> Observable
        self._species_info = {}  # SBML species ID -> _SpeciesInfo
        self._cpt_size = {}  # PySB compartment name -> size Parameter
        self._func_defs = {}  # SBML function definition ID -> (args, body)
        self._boundary_species = set()  # SBML species IDs with boundaryCondition=True
        self._obs_init = {}  # Observable -> initial-amount Parameter
        self._pending = set()  # symbols whose initial value is not yet known
        # SBML ID -> (value Parameter, symbol that is pending until resolved,
        # compartment size converting concentrations to amounts or None,
        # whether an initial assignment gives a concentration)
        self._init_targets = {}
        self._raw_initial = {}  # species ID -> (value, is_amount) from attributes
        self._simple_functions = _simple_functions()
        self._relational_functions = _relational_functions()

        self._reserve_names(sbml_model)

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

        self._sym["time"] = pysb_time
        self._parse_function_definitions(sbml_model)
        self._parse_compartments(sbml_model)
        self._parse_species(sbml_model)
        self._parse_parameters(sbml_model)
        self._parse_rate_rule_variables(sbml_model)
        self._parse_assignment_rules(sbml_model)
        self._parse_initial_assignments(sbml_model)
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

    # ------------------------------------------------------------------
    # Names
    # ------------------------------------------------------------------

    def _reserve_names(self, sbml_model):
        """Reserve the sanitised name of every SBML id in the document.

        Names generated by the importer must not collide with these.
        """
        self._reserved = set()
        self._used = set()
        for getter, count in (
            (sbml_model.getCompartment, sbml_model.getNumCompartments()),
            (sbml_model.getSpecies, sbml_model.getNumSpecies()),
            (sbml_model.getParameter, sbml_model.getNumParameters()),
            (sbml_model.getReaction, sbml_model.getNumReactions()),
            (sbml_model.getFunctionDefinition, sbml_model.getNumFunctionDefinitions()),
        ):
            for i in range(count):
                if getter(i).getId():
                    self._reserved.add(_sanitize_id(getter(i).getId()))

        self._local_id_counts = Counter()
        for i in range(sbml_model.getNumReactions()):
            kl = sbml_model.getReaction(i).getKineticLaw()
            if kl is not None:
                for j in range(kl.getNumParameters()):
                    self._local_id_counts[kl.getParameter(j).getId()] += 1

    def _entity_name(self, sbml_id):
        """Claim the PySB component name for an SBML id"""
        name = _sanitize_id(sbml_id)
        self._used.add(name)
        self._id_map[sbml_id] = name
        return name

    def _generated_name(self, base):
        """Return a component name based on *base* that is not yet in use"""
        name = _sanitize_id(base)
        while name in self._reserved or name in self._used:
            name += "_"
        self._used.add(name)
        return name

    # ------------------------------------------------------------------
    # Math
    # ------------------------------------------------------------------

    def _parse_function_definitions(self, sbml_model):
        """Record SBML FunctionDefinition elements for inline expansion.

        Each definition is stored in ``self._func_defs`` as its formal
        argument names and body, and is expanded wherever it is called.
        """
        for i in range(sbml_model.getNumFunctionDefinitions()):
            fd = sbml_model.getFunctionDefinition(i)
            fd_math = fd.getMath()

            if fd_math is None or fd_math.getType() != libsbml.AST_LAMBDA:
                continue

            # Children are formal arguments followed by the body expression
            n_children = fd_math.getNumChildren()
            if n_children == 0:
                continue
            arg_names = [fd_math.getChild(j).getName() for j in range(n_children - 1)]
            self._func_defs[fd.getId()] = (arg_names, fd_math.getChild(n_children - 1))

    def _ast_to_sympy(self, node, local):
        """Convert a libsbml ASTNode to a sympy expression.

        *local* maps names that take precedence over model-level symbols
        (function arguments, local kinetic-law parameters).
        """
        node_type = node.getType()
        children = [node.getChild(j) for j in range(node.getNumChildren())]

        def convert(child):
            return self._ast_to_sympy(child, local)

        if node_type == libsbml.AST_INTEGER:
            return sympy.Integer(node.getInteger())
        if node_type in (libsbml.AST_REAL, libsbml.AST_REAL_E):
            return sympy.Float(node.getReal())
        if node_type == libsbml.AST_RATIONAL:
            return sympy.Rational(node.getNumerator(), node.getDenominator())
        if node_type == libsbml.AST_NAME_TIME:
            return pysb_time
        if node_type == libsbml.AST_NAME_AVOGADRO:
            return sympy.Float(6.02214076e23)
        if node_type == libsbml.AST_NAME:
            name = node.getName()
            if name in local:
                return local[name]
            if name in self._sym:
                return self._sym[name]
            raise _UnsupportedMath('unknown or unsupported symbol "{}"'.format(name))
        if node_type == libsbml.AST_CONSTANT_E:
            return sympy.Float(math.e)
        if node_type == libsbml.AST_CONSTANT_PI:
            return sympy.Float(math.pi)
        if node_type == libsbml.AST_CONSTANT_TRUE:
            return sympy.true
        if node_type == libsbml.AST_CONSTANT_FALSE:
            return sympy.false

        args = [convert(c) for c in children]

        if node_type == libsbml.AST_PLUS:
            return sympy.Add(*args)
        if node_type == libsbml.AST_TIMES:
            return sympy.Mul(*args)
        if node_type == libsbml.AST_MINUS:
            return -args[0] if len(args) == 1 else args[0] - sympy.Add(*args[1:])
        if node_type == libsbml.AST_DIVIDE:
            return args[0] / args[1]
        if node_type in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
            exponent = args[1]
            if isinstance(exponent, sympy.Float) and float(exponent).is_integer():
                exponent = sympy.Integer(int(exponent))
            return args[0] ** exponent
        if node_type in self._simple_functions:
            return self._simple_functions[node_type](*args)
        if node_type == libsbml.AST_FUNCTION_LOG:
            # <log/> without <logbase> is base 10; otherwise the first child
            # is the base
            if len(args) == 1:
                return sympy.log(args[0], 10)
            return sympy.log(args[1], args[0])
        if node_type == libsbml.AST_FUNCTION_ROOT:
            # <root/> without <degree> is a square root
            if len(args) == 1:
                return sympy.sqrt(args[0])
            return sympy.root(args[1], args[0])
        if node_type == libsbml.AST_FUNCTION_PIECEWISE:
            pieces = [(args[j], args[j + 1]) for j in range(0, len(args) - 1, 2)]
            if len(args) % 2:
                pieces.append((args[-1], sympy.true))
            return sympy.Piecewise(*pieces)
        if node_type in self._relational_functions:
            relation = self._relational_functions[node_type]
            return sympy.And(*[relation(a, b) for a, b in zip(args, args[1:])])
        if node_type == libsbml.AST_LOGICAL_AND:
            return sympy.And(*args)
        if node_type == libsbml.AST_LOGICAL_OR:
            return sympy.Or(*args)
        if node_type == libsbml.AST_LOGICAL_XOR:
            return sympy.Xor(*args)
        if node_type == libsbml.AST_LOGICAL_NOT:
            return sympy.Not(args[0])
        if node_type == libsbml.AST_FUNCTION_MAX:
            return sympy.Max(*args)
        if node_type == libsbml.AST_FUNCTION_MIN:
            return sympy.Min(*args)
        if node_type == libsbml.AST_FUNCTION_QUOTIENT:
            return sympy.floor(args[0] / args[1])
        if node_type == libsbml.AST_FUNCTION_REM:
            return args[0] - args[1] * sympy.floor(args[0] / args[1])
        if node_type == libsbml.AST_FUNCTION:
            name = node.getName()
            if name not in self._func_defs:
                raise _UnsupportedMath('unknown function "{}"'.format(name))
            arg_names, body = self._func_defs[name]
            if len(arg_names) != len(args):
                raise _UnsupportedMath(
                    'function "{}" called with the wrong number of arguments'.format(
                        name
                    )
                )
            return self._ast_to_sympy(body, dict(zip(arg_names, args)))

        raise _UnsupportedMath(
            "unsupported MathML construct ({})".format(
                libsbml.formulaToL3String(node) or node_type
            )
        )

    def _formula_to_sympy(self, ast_node, local=None):
        """Convert a libsbml ASTNode to a sympy expression.

        Returns ``sympy.Integer(0)`` when the node cannot be converted (after
        calling :meth:`_warn_or_except` so the failure is visible to the
        caller).
        """
        try:
            return self._ast_to_sympy(ast_node, local or {})
        except _UnsupportedMath as exc:
            self._warn_or_except(
                'Could not convert SBML math "{}": {}'.format(
                    libsbml.formulaToL3String(ast_node), exc
                )
            )
            return sympy.Integer(0)

    # ------------------------------------------------------------------
    # Compartments, species, parameters
    # ------------------------------------------------------------------

    def _parse_compartments(self, sbml_model):
        """Create PySB Compartments and matching size Parameters.

        Each SBML ``<compartment>`` becomes a :class:`~pysb.core.Compartment`
        plus a :class:`~pysb.core.Parameter` named ``<id>_size`` holding the
        initial size.  The ``outside`` attribute is mapped to a PySB parent
        when the compartment dimensions differ (BioNetGen does not allow
        nesting compartments of equal dimension, which is how SBML Level 2
        commonly arranges volumes).  Parents are created before children,
        regardless of document order.
        """
        compartments = {}
        for i in range(sbml_model.getNumCompartments()):
            cpt = sbml_model.getCompartment(i)
            compartments[cpt.getId()] = cpt

        def dimension(cpt):
            dims = cpt.getSpatialDimensionsAsDouble()
            return int(dims) if dims == dims else 3  # NaN: unset (L3)

        self._compartments_by_id = compartments
        pending = list(compartments.values())
        while pending:
            ready = [
                c
                for c in pending
                if not c.isSetOutside()
                or c.getOutside() not in compartments
                or c.getOutside() in self._id_map
            ]
            if not ready:  # circular 'outside' references
                ready = pending[:1]
            for cpt in ready:
                pending.remove(cpt)
                self._create_compartment(cpt, dimension(cpt), dimension)

    def _create_compartment(self, cpt, dims, dimension_of):
        cpt_id = cpt.getId()
        cpt_name = self._entity_name(cpt_id)

        size_val = cpt.getSize() if cpt.isSetSize() else 1.0
        if size_val != size_val:  # NaN
            size_val = 1.0
        size_param = self.parameter(
            self._generated_name(cpt_name + "_size"), size_val, nonnegative=None
        )
        self._sym[cpt_id] = size_param
        self._cpt_size[cpt_name] = size_param
        self._init_targets[cpt_id] = (size_param, size_param, None, False)

        parent = None
        if cpt.isSetOutside():
            outside = cpt.getOutside()
            outer = self._compartments_by_id.get(outside)
            if (
                outer is not None
                and outside in self._id_map
                and dimension_of(outer) != dims
            ):
                parent = self.model.compartments.get(self._id_map[outside])

        self.compartment(name=cpt_name, parent=parent, dimension=dims, size=size_param)

    def _parse_species(self, sbml_model):
        """Create Monomers, Observables, and initial conditions for SBML species.

        Each ``<species>`` element becomes:

        * A site-less :class:`~pysb.core.Monomer`.
        * An :class:`~pysb.core.Observable` ``obs_<id>`` whose value is the
          species amount.
        * An initial-condition :class:`~pysb.core.Parameter` ``<id>_0``
          holding the initial **amount**: ``initialAmount`` as given (it takes
          precedence when both are present, per the SBML specification), or
          ``initialConcentration`` multiplied by the compartment size.

        In SBML math the species symbol is the observable (amount) when the
        species has ``hasOnlySubstanceUnits="true"`` or lives in a
        zero-dimensional compartment, and ``observable / size`` otherwise.

        Species with ``boundaryCondition="true"`` are created as *fixed*
        initials so their amount is held constant throughout simulation,
        **unless** a rate rule governs them; in that case the rate rule drives
        their dynamics and the initial should not be fixed.  Species defined
        by an assignment rule become Expressions instead.
        """
        for i in range(sbml_model.getNumSpecies()):
            sp = sbml_model.getSpecies(i)
            sp_id = sp.getId()
            sp_name = self._entity_name(sp_id)

            if sp_id in self._assigned_vars:
                # The amount is defined by an assignment rule: no Monomer is
                # needed; the species is an Expression (see
                # _parse_assignment_rules) and reactions do not change it.
                continue

            mon = self.monomer(sp_name)

            cpt_id = sp.getCompartment()
            cpt = (
                self.model.compartments.get(self._id_map.get(cpt_id, ""))
                if cpt_id
                else None
            )

            cp = ComplexPattern([MonomerPattern(mon, {}, cpt)], cpt)

            # Observable for use in kinetic law and assignment rule expressions
            obs = self.observable(self._generated_name("obs_" + sp_name), cp)
            self._species_obs[sp_id] = obs

            size = self._cpt_size.get(cpt.name) if cpt is not None else None
            if size is not None and cpt.dimension == 0:
                size = None  # concentration and amount coincide
            is_concentration = size is not None and not sp.getHasOnlySubstanceUnits()
            self._species_info[sp_id] = (cpt, size if is_concentration else None)
            self._sym[sp_id] = obs / size if is_concentration else obs

            init_param = self.parameter(
                self._generated_name(sp_name + "_0"), 0.0, nonnegative=None
            )
            self._obs_init[obs] = init_param
            self._pending.add(obs)
            self._init_targets[sp_id] = (init_param, obs, size, is_concentration)
            if sp.isSetInitialAmount():
                self._raw_initial[sp_id] = (sp.getInitialAmount(), True)
            elif sp.isSetInitialConcentration():
                self._raw_initial[sp_id] = (sp.getInitialConcentration(), False)
            else:
                self._raw_initial[sp_id] = (0.0, True)

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
        and :meth:`_parse_rate_rules` respectively.  Parameters without a
        value are created with a placeholder value if an initial assignment
        supplies one; otherwise a warning is issued and they are not created.
        """
        assigned = {
            sbml_model.getInitialAssignment(i).getSymbol()
            for i in range(sbml_model.getNumInitialAssignments())
        }
        for i in range(sbml_model.getNumParameters()):
            param = sbml_model.getParameter(i)
            param_id = param.getId()

            # Parameters governed by rules are handled separately
            if param_id in self._assigned_vars or param_id in self._rate_rule_vars:
                self._entity_name(param_id)
                continue

            param_name = self._entity_name(param_id)
            has_value = param.isSetValue() and param.getValue() == param.getValue()
            if not has_value and param_id not in assigned:
                warnings.warn(
                    'Parameter "{}" has neither a value nor an initial '
                    "assignment; it is not defined in the PySB model".format(param_id)
                )
                continue

            pysb_param = self.parameter(
                param_name, param.getValue() if has_value else 0.0, nonnegative=None
            )
            self._sym[param_id] = pysb_param
            self._init_targets[param_id] = (pysb_param, pysb_param, None, False)

    def _expression_value(self, expr):
        """Evaluate *expr* at t=0, or None if it depends on unresolved values"""
        expr = expr.subs(
            [(a, a.expand_expr()) for a in expr.atoms() if isinstance(a, Expression)]
        )
        values = {}
        for sym in expr.free_symbols:
            if sym in self._pending:
                return None
            if isinstance(sym, Parameter):
                values[sym] = sym.value
            elif isinstance(sym, Observable) and sym in self._obs_init:
                values[sym] = self._obs_init[sym].value
            elif sym is pysb_time:
                values[sym] = 0.0
            else:
                return None
        value = expr.subs(values)
        if value.is_number and not value.has(sympy.nan, sympy.zoo):
            return float(value)
        return None

    def _resolve_initial(self, target, value, is_amount):
        """Store the initial *value* of an SBML entity if it can be resolved.

        A value that is a concentration is converted to an amount with the
        compartment size, which must itself be resolved first.  Returns False
        if the value cannot be stored yet.
        """
        param, pending_symbol, size, _ = self._init_targets[target]
        if not is_amount and size is not None:
            if size in self._pending:
                return False
            value *= size.value
        param.value = value
        self._pending.discard(pending_symbol)
        return True

    def _parse_initial_assignments(self, sbml_model):
        """Resolve initial values, including SBML initialAssignment elements.

        Initial assignments (the mechanism used by PySB's SBML exporter for
        species) override ``value`` / ``initialAmount`` /
        ``initialConcentration`` and may refer to anything with an initial
        value, including each other and assignment-rule variables.  They are
        evaluated numerically, repeating until no further progress is made.
        Species initial concentrations are converted to amounts with the
        final compartment size.
        """
        items = []  # (target, value or sympy expression, is_assignment)
        assigned = set()
        for i in range(sbml_model.getNumInitialAssignments()):
            ia = sbml_model.getInitialAssignment(i)
            target = ia.getSymbol()
            if ia.getMath() is None:
                continue
            if target not in self._init_targets:
                self._warn_or_except(
                    'Initial assignment to "{}" is not supported (not a '
                    "species, parameter or compartment)".format(target)
                )
                continue
            # The target counts as unresolved until the assignment has been
            # evaluated, as it overrides any value set on the entity itself.
            self._pending.add(self._init_targets[target][1])
            assigned.add(target)
            items.append((target, self._formula_to_sympy(ia.getMath()), True))
        for sp_id, (value, is_amount) in self._raw_initial.items():
            if sp_id not in assigned:
                items.append((sp_id, (value, is_amount), False))

        progress = True
        while items and progress:
            progress = False
            for item in list(items):
                target, payload, is_assignment = item
                if is_assignment:
                    value = self._expression_value(payload)
                    if value is None:
                        continue
                    # An assigned value is a concentration for species in
                    # concentration units, an amount otherwise
                    is_amount = not self._init_targets[target][3]
                else:
                    value, is_amount = payload
                if self._resolve_initial(target, value, is_amount):
                    items.remove(item)
                    progress = True

        for target, _, _ in items:
            self._warn_or_except(
                'Could not evaluate the initial value of "{}" (circular or '
                "unsupported dependencies)".format(target)
            )

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def _parse_assignment_rules(self, sbml_model):
        """Create PySB Expressions for SBML assignmentRule elements.

        Each rule ``variable = f(...)`` becomes an
        :class:`~pysb.core.Expression` whose value is recomputed at every
        time point.  Rules are processed in dependency order, so a rule may
        refer to rules defined later in the document.  Rules targeting
        compartment IDs (dynamic compartment sizes) are not supported and
        trigger :meth:`_warn_or_except`.  A species defined by an assignment
        rule has no Monomer; it is an Expression like any other rule
        variable.
        """
        cpt_ids = {
            sbml_model.getCompartment(i).getId()
            for i in range(sbml_model.getNumCompartments())
        }

        rules = {}
        for i in range(sbml_model.getNumRules()):
            rule = sbml_model.getRule(i)
            if rule.getTypeCode() != libsbml.SBML_ASSIGNMENT_RULE:
                continue
            var_id = rule.getVariable()
            if var_id in cpt_ids:
                self._warn_or_except(
                    'Assignment rule targeting compartment "{}" (dynamic '
                    "compartment size) is not supported and will be "
                    "ignored".format(var_id)
                )
                continue
            if var_id in self._rate_rule_vars:
                # Rate-rule variables are already model components
                continue
            if rule.getMath() is not None:
                rules[var_id] = rule

        # Order by dependency (depth-first), so referenced rules come first
        ordered = []
        state = {}

        def visit(var_id):
            if state.get(var_id) == "done":
                return
            if state.get(var_id) == "visiting":
                self._warn_or_except(
                    'Circular dependency between assignment rules at "{}"'.format(
                        var_id
                    )
                )
                return
            state[var_id] = "visiting"
            for name in sorted(_ast_names(rules[var_id].getMath())):
                if name in rules:
                    visit(name)
            state[var_id] = "done"
            ordered.append(var_id)

        for var_id in rules:
            visit(var_id)

        for var_id in ordered:
            var_name = self._entity_name(var_id)
            expr_val = self._formula_to_sympy(rules[var_id].getMath())
            self._sym[var_id] = self.expression(var_name, expr_val)

    def _parse_rate_rule_variables(self, sbml_model):
        """Create Monomers, Observables and Initials for rate-rule parameters.

        Rate-rule variables that are SBML parameters (not species) are
        modelled as site-less Monomers whose amount is the parameter value.
        If the model has compartments they are placed in the first one so
        that PySB patterns are concrete; this is a formal assignment that
        does not affect the mathematics.
        """
        self._default_cpt = next(iter(self.model.compartments.values()), None)

        cpt_ids = set(self._compartments_by_id)
        for var_id in sorted(self._rate_rule_vars):
            if var_id in self._species_obs:
                continue

            if var_id in cpt_ids:
                self._warn_or_except(
                    'Rate rule targeting compartment "{}" (dynamic compartment '
                    "size) is not supported and will be ignored".format(var_id)
                )
                continue

            var_name = self._entity_name(var_id)
            param = sbml_model.getParameter(var_id)
            init_val = param.getValue() if (param and param.isSetValue()) else 0.0

            mon = self.monomer(var_name)
            cpt = self._default_cpt
            cp = ComplexPattern([MonomerPattern(mon, {}, cpt)], cpt)
            obs = self.observable(self._generated_name("obs_" + var_name), cp)
            self._species_obs[var_id] = obs
            self._sym[var_id] = obs

            init_p = self.parameter(
                self._generated_name(var_name + "_0"), init_val, nonnegative=None
            )
            self._obs_init[obs] = init_p
            self._init_targets[var_id] = (init_p, obs, None, False)
            self.initial(cp, init_p)

    def _parse_rate_rules(self, sbml_model):
        """Handle SBML rateRule elements.

        Each rate-rule variable X is a species (Monomer with no sites) and
        its ODE is encoded as a synthesis rule::

            None >> X()   with rate = RHS / compartment size

        BioNetGen multiplies the rate of a synthesis rule by the size of the
        product compartment, so the rule contributes ``dN/dt`` = RHS to the
        amount of X.  For a species in concentration units the SBML rule
        defines ``d[X]/dt``, so the RHS is first multiplied by the
        compartment size, which cancels the BioNetGen scaling.  The amount
        can go negative (e.g. membrane voltage), which is valid for
        ScipyOdeSimulator.
        """
        processed_rate_vars = set()
        for i in range(sbml_model.getNumRules()):
            rule = sbml_model.getRule(i)
            if rule.getTypeCode() != libsbml.SBML_RATE_RULE:
                continue

            var_id = rule.getVariable()

            # Skip duplicates: only the first rule for each variable is used.
            # Also skip variables that could not be created (compartments).
            if var_id in processed_rate_vars or var_id not in self._species_obs:
                continue
            processed_rate_vars.add(var_id)

            if rule.getMath() is None:
                continue
            rate_expr = self._formula_to_sympy(rule.getMath())

            if var_id in self._species_info:
                cpt, conc_size = self._species_info[var_id]
            else:
                cpt, conc_size = self._default_cpt, None
            bng_size = self._cpt_size[cpt.name] if cpt is not None else 1
            amount_rate = rate_expr * conc_size if conc_size is not None else rate_expr
            var_name = self._id_map[var_id]

            rate_comp = self._expr_or_param(
                sympy.cancel(amount_rate / bng_size), var_name + "_rate"
            )
            cp = ComplexPattern(
                [MonomerPattern(self.model.monomers[var_name], {}, cpt)], cpt
            )
            self.rule(
                self._generated_name(var_name + "_ode"),
                RuleExpression(
                    ReactionPattern([]),  # null (no reactants)
                    ReactionPattern([cp]),  # >> X()
                    is_reversible=False,
                ),
                rate_comp,
            )

    def _expr_or_param(self, expr, name):
        """Return a bare Parameter if *expr* is one, else a new Expression.

        Many simple mass-action rates reduce to a single PySB
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
        if isinstance(expr, Parameter) and expr in self.model.parameters.values():
            return expr
        return self.expression(self._generated_name(name), expr)

    # ------------------------------------------------------------------
    # Reactions
    # ------------------------------------------------------------------

    def _local_parameters(self, rxn, rxn_name):
        """Create Parameters for the local parameters of a reaction's law.

        Local parameters (``<listOfParameters>`` in Level 2,
        ``<listOfLocalParameters>`` in Level 3) take precedence over global
        symbols within the kinetic law.  Each becomes a model Parameter, named
        by its SBML id if that is unique in the model and
        ``<reaction>_<id>`` otherwise.

        Returns
        -------
        dict
            SBML id -> Parameter, for use as the local namespace of the law
        """
        local = {}
        kl = rxn.getKineticLaw()
        for j in range(kl.getNumParameters()):
            lp = kl.getParameter(j)
            lp_id = lp.getId()
            if not lp.isSetValue():
                self._warn_or_except(
                    'Local parameter "{}" of reaction "{}" has no value'.format(
                        lp_id, rxn.getId()
                    )
                )
                continue
            base = _sanitize_id(lp_id)
            if self._local_id_counts[lp_id] == 1 and base not in self._reserved:
                name = self._generated_name(base)
            else:
                name = self._generated_name("{}_{}".format(rxn_name, base))
            local[lp_id] = self.parameter(name, lp.getValue(), nonnegative=None)
        return local

    def _stoichiometry(self, refs):
        """Non-boundary species of a reactant/product list.

        Returns
        -------
        tuple of (dict, dict)
            Pattern copies (integer) and exact stoichiometry (float) per
            SBML species id.  Boundary species are **excluded**: per the
            SBML specification their amounts are not changed by reactions,
            and including them in PySB rule patterns would make BNG change
            them in the ODEs.
        """
        copies, exact = {}, {}
        for j in range(refs.size()):
            sr = refs.get(j)
            sp_id = sr.getSpecies()
            if sp_id in self._boundary_species or sp_id not in self._species_info:
                continue
            stoich = sr.getStoichiometry() if sr.isSetStoichiometry() else 1.0
            if stoich != stoich:  # NaN: unset (L3)
                stoich = 1.0
            exact[sp_id] = exact.get(sp_id, 0.0) + stoich
            copies[sp_id] = copies.get(sp_id, 0) + _copies(stoich)
        return {sp: n for sp, n in copies.items() if n > 0}, exact

    def _species_pattern(self, sp_id):
        cpt = self._species_info[sp_id][0]
        mon = self.model.monomers[self._id_map[sp_id]]
        return ComplexPattern([MonomerPattern(mon, {}, cpt)], cpt)

    def _patterns(self, copies):
        return [
            self._species_pattern(sp_id)
            for sp_id, n in copies.items()
            for _ in range(n)
        ]

    def _bng_rate_scale(self, reactants, products):
        """The factor BioNetGen applies to a rule rate for compartment sizes.

        Mirrors ``get_intensive_to_extensive_units_conversion`` in BNG's
        ``Rxn.pm``: reactions of order two or more are divided by the sizes
        of all reactant compartments except one anchor (a surface if there
        is one, otherwise a volume), and synthesis is multiplied by the
        product compartment size.
        """
        reactant_cpts = [
            self._species_info[sp][0] for sp, n in reactants.items() for _ in range(n)
        ]
        reactant_cpts = [c for c in reactant_cpts if c is not None]
        if reactant_cpts:
            surfaces = [c for c in reactant_cpts if c.dimension == 2]
            volumes = [c for c in reactant_cpts if c.dimension == 3]
            if surfaces:
                surfaces.pop(0)
            elif volumes:
                volumes.pop(0)
            scale = sympy.Integer(1)
            for cpt in surfaces + volumes:
                scale = scale / self._cpt_size[cpt.name]
            return scale

        product_cpts = {
            self._species_info[sp][0] for sp, n in products.items() if n > 0
        } - {None}
        if len(product_cpts) == 1:
            return self._cpt_size[product_cpts.pop().name]
        return sympy.Integer(1)

    def _mass_action_rate(self, flux, reactants, products):
        """Rate of a rule ``reactants >> products`` that carries *flux*.

        Returns None if the required division by the reactant amounts is
        undefined when a reactant amount is zero.
        """
        observables = [self._species_obs[sp] for sp in reactants]
        divisor = sympy.Integer(1)
        for sp, n in reactants.items():
            divisor *= self._species_obs[sp] ** n

        symmetry = 1
        for sp, n in reactants.items():
            kept = min(n, products.get(sp, 0))
            symmetry *= math.factorial(kept) * math.factorial(n - kept)

        rate = symmetry * flux / (self._bng_rate_scale(reactants, products) * divisor)
        if _unsafe_quotient(rate, observables):
            return None
        return rate

    def _rule_patterns(self, reactants, products):
        """Reactant and product patterns for a rule.

        The kept copies of each reactant species come first among the
        products, in reactant order, so that BioNetGen's positional mapping
        of reactants to products treats them as unchanged.
        """
        kept = {sp: min(n, products.get(sp, 0)) for sp, n in reactants.items()}
        extra = {sp: n - kept.get(sp, 0) for sp, n in products.items()}
        return self._patterns(reactants), self._patterns(kept) + self._patterns(extra)

    def _add_rule(self, name, reactants, products, rate_expr, rate_name):
        reactant_pats, product_pats = self._rule_patterns(reactants, products)
        self.rule(
            name,
            RuleExpression(
                ReactionPattern(reactant_pats),
                ReactionPattern(product_pats),
                is_reversible=False,
            ),
            self._expr_or_param(rate_expr, rate_name),
        )

    def _add_signed_synthesis_rules(self, rxn_name, flux, net):
        """Encode a reaction as one synthesis rule per changed species.

        Each rule ``None >> S()`` contributes ``rate * size`` to the amount of
        S in the ODEs (BioNetGen's zero-order scaling), so the rate is
        ``net stoichiometry * flux / size``.  Valid for any kinetic law.
        """
        flux_comp = None
        for sp_id, stoich in net.items():
            if stoich == 0:
                continue
            if flux_comp is None:
                flux_comp = self.expression(
                    self._generated_name(rxn_name + "_flux"), flux
                )
            cpt = self._species_info[sp_id][0]
            size = self._cpt_size[cpt.name] if cpt is not None else 1
            coeff = (
                sympy.Integer(int(stoich))
                if stoich == int(stoich)
                else sympy.Float(stoich)
            )
            sp_name = self._id_map[sp_id]
            self.rule(
                self._generated_name("{}_{}".format(rxn_name, sp_name)),
                RuleExpression(
                    ReactionPattern([]),
                    ReactionPattern([self._species_pattern(sp_id)]),
                    is_reversible=False,
                ),
                self.expression(
                    self._generated_name("{}_{}_rate".format(rxn_name, sp_name)),
                    coeff * flux_comp / size,
                ),
            )

    def _parse_reactions(self, sbml_model):
        """Create PySB Rules for SBML reaction elements.

        An SBML kinetic law is the flux of the reaction in amount per time.
        A PySB rule ``reactants >> products`` with a function-based rate
        contributes ``rate * prod(reactant amounts)`` to the ODEs after
        BioNetGen has applied its own symmetry and compartment-size scaling
        (see the module documentation).  The importer therefore sets

        .. math::

            \\text{rate} = \\frac{\\text{symmetry} \\cdot J}
                                {\\text{scale} \\cdot \\prod_i S_i^{s_i}}

        For reversible reactions the net-flux law is split into positive and
        negative additive terms by :func:`_split_reversible_rate`, giving
        rules ``<id>_fwd`` and ``<id>_rev``.

        Reactions are encoded as signed synthesis rules instead (see
        :meth:`_add_signed_synthesis_rules`) when they involve more than one
        compartment, when the division above is undefined for zero reactant
        amounts, or when a reversible law cannot be split.

        The ``fast`` attribute (quasi-steady-state flag, deprecated in SBML
        Level 3 Version 2) triggers a :mod:`warnings` warning and is otherwise
        ignored.  Reactions without an ``id`` attribute receive the fallback
        name ``r{i}``.
        """
        for i in range(sbml_model.getNumReactions()):
            rxn = sbml_model.getReaction(i)
            rxn_id = rxn.getId()
            if rxn_id:
                rxn_name = self._entity_name(rxn_id)
            else:
                rxn_name = self._generated_name("r{}".format(i))

            # The SBML 'fast' attribute signals quasi-steady-state (QSS)
            # treatment.  PySB has no QSS mechanism, so warn if set.
            if rxn.isSetFast() and rxn.getFast():
                warnings.warn(
                    'Reaction "{}" has fast="true" (quasi-steady-state); '
                    "this attribute has no effect in PySB and is "
                    "ignored.".format(rxn_id)
                )

            kl = rxn.getKineticLaw()
            if kl is None:
                self._warn_or_except("Reaction {} has no kinetic law".format(rxn_id))
                continue

            local = self._local_parameters(rxn, rxn_name)
            flux = self._formula_to_sympy(kl.getMath(), local)

            reactants, reactants_exact = self._stoichiometry(rxn.getListOfReactants())
            products, products_exact = self._stoichiometry(rxn.getListOfProducts())
            if not reactants_exact and not products_exact:
                continue  # only boundary species: no effect on the ODEs

            net = {
                sp: products_exact.get(sp, 0.0) - reactants_exact.get(sp, 0.0)
                for sp in {**reactants_exact, **products_exact}
            }

            compartments = {
                self._species_info[sp][0].name
                if self._species_info[sp][0] is not None
                else None
                for sp in {**reactants_exact, **products_exact}
            }
            integral = _is_integral(reactants_exact, reactants) and _is_integral(
                products_exact, products
            )
            if (
                integral
                and len(compartments) <= 1
                and self._add_mass_action_rules(
                    rxn, rxn_name, flux, reactants, products
                )
            ):
                continue
            self._add_signed_synthesis_rules(rxn_name, flux, net)

    def _add_mass_action_rules(self, rxn, rxn_name, flux, reactants, products):
        """Add rules for a reaction; return False if this is not possible"""
        if not rxn.getReversible():
            rate = self._mass_action_rate(flux, reactants, products)
            if rate is None:
                return False
            self._add_rule(rxn_name, reactants, products, rate, rxn_name + "_rate")
            return True

        fwd_flux, rev_flux = _split_reversible_rate(flux)
        if fwd_flux is None:
            return False
        fwd_rate = self._mass_action_rate(fwd_flux, reactants, products)
        rev_rate = self._mass_action_rate(rev_flux, products, reactants)
        if fwd_rate is None or rev_rate is None:
            return False
        self._add_rule(
            self._generated_name(rxn_name + "_fwd"),
            reactants,
            products,
            fwd_rate,
            rxn_name + "_fwd_rate",
        )
        self._add_rule(
            self._generated_name(rxn_name + "_rev"),
            products,
            reactants,
            rev_rate,
            rxn_name + "_rev_rate",
        )
        return True


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
    # Distribute factors over a sum (volume * (forward - reverse)), but do not
    # expand the terms themselves, which can be huge for large kinetic laws
    terms = sympy.Add.make_args(
        sympy.expand(
            expr, deep=False, multinomial=False, power_base=False, power_exp=False
        )
    )

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
        when ``use_libsbml=False`` (e.g. ``atomize=True``). Passing any with
        ``use_libsbml=True`` raises :class:`TypeError`, rather than silently
        ignoring them.

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
        if kwargs:
            raise TypeError(
                "Unexpected keyword argument(s) for the libsbml importer: {}. "
                "Options such as atomize apply to the legacy Atomizer "
                "backend; pass use_libsbml=False to use it.".format(
                    ", ".join(sorted(kwargs))
                )
            )
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
