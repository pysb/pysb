"""
Module containing a class for exporting a PySB model to SBML using libSBML

For information on how to use the model exporters, see the documentation
for :py:mod:`pysb.export`.
"""
import pysb
import pysb.bng
from pysb.export import Exporter
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
            'Error encountered converting to SBML. '
            'LibSBML returned error code {}: "{}"'.format(
                value,
                libsbml.OperationReturnValue_toString(value).strip()
            )
        )
    elif value is None:
        raise ValueError('LibSBML returned a null value')


def _add_ci(x_doc, x_parent, name):
    """ Add <ci>name</ci> element to <x_parent> within x_doc """
    ci = x_doc.createElement('ci')
    ci.appendChild(x_doc.createTextNode(name))
    x_parent.appendChild(ci)


def _xml_to_ast(x_element):
    """ Wrap MathML fragment with <math> tag and convert to libSBML AST """
    x_doc = Document()
    x_mathml = x_doc.createElement('math')
    x_mathml.setAttribute('xmlns', 'http://www.w3.org/1998/Math/MathML')

    x_mathml.appendChild(x_element)
    x_doc.appendChild(x_mathml)

    mathml_ast = libsbml.readMathMLFromString(x_doc.toxml())
    _check(mathml_ast)
    return mathml_ast


def _mathml_expr_call(expr):
    """ Generate an XML <apply> expression call """
    x_doc = Document()
    x_apply = x_doc.createElement('apply')
    x_doc.appendChild(x_apply)

    _add_ci(x_doc, x_apply, expr.name)
    for sym in expr.expand_expr(expand_observables=True).free_symbols:
        if isinstance(sym, pysb.Expression):
            continue
        _add_ci(x_doc, x_apply, sym.name if isinstance(sym, pysb.Parameter) else str(sym))

    return x_apply


class SbmlExporter(Exporter):
    """A class for returning the SBML for a given PySB model.

    Inherits from :py:class:`pysb.export.Exporter`, which implements
    basic functionality for all exporters.
    """
    def __init__(self, *args, **kwargs):
        if not libsbml:
            raise ImportError('The SbmlExporter requires the libsbml python package')
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
            The SBML level and version to use. The default is SBML level 3, version 2. Conversion
            to other levels/versions may not be possible or may lose fidelity.

        Returns
        -------
        libsbml.SBMLDocument
            A libSBML document converted form the PySB model
        """
        doc = libsbml.SBMLDocument(3, 2)
        smodel = doc.createModel()
        _check(smodel)
        _check(smodel.setName(self.model.name))

        pysb.bng.generate_equations(self.model)

        # Docstring
        if self.docstring:
            notes_str = """
            <notes>
                <body xmlns="http://www.w3.org/1999/xhtml">
                    <p>%s</p>
                </body>
            </notes>""" % self.docstring.replace("\n", "<br />\n"+" "*20)
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
            _check(c.setId('default'))
            _check(c.setSpatialDimensions(3))
            _check(c.setSize(1))
            _check(c.setConstant(True))

        # Expressions
        for expr in itertools.chain(
                self.model.expressions_constant(),
                self.model.expressions_dynamic(include_local=False),
                self.model._derived_expressions
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

            expr_mathml = self._sympy_to_sbmlast(expr.expand_expr(expand_observables=True))
            _check(expr_rule.setMath(expr_mathml))

        # Initial values/assignments
        fixed_species_idx = set()
        initial_species_idx = set()
        for ic in self.model.initials:
            sp_idx = self.model.get_species_index(ic.pattern)
            ia = smodel.createInitialAssignment()
            _check(ia)
            _check(ia.setSymbol('__s{}'.format(sp_idx)))
            init_mathml = self._sympy_to_sbmlast(Symbol(ic.value.name))
            _check(ia.setMath(init_mathml))
            initial_species_idx.add(sp_idx)

            if ic.fixed:
                fixed_species_idx.add(sp_idx)

        # Species
        for i, s in enumerate(self.model.species):
            sp = smodel.createSpecies()
            _check(sp)
            _check(sp.setId('__s{}'.format(i)))
            if self.model.compartments:
                # Try to determine compartment, which must be unique for the species
                mon_cpt = set(mp.compartment for mp in s.monomer_patterns if mp.compartment is not None)
                if len(mon_cpt) == 0 and s.compartment:
                    compartment_name = s.compartment_name
                elif len(mon_cpt) == 1:
                    mon_cpt = mon_cpt.pop()
                    if s.compartment is not None and mon_cpt != s.compartment:
                        raise ValueError('Species {} has different monomer and species compartments, '
                                         'which is not supported in SBML'.format(s))
                    compartment_name = mon_cpt.name
                else:
                    raise ValueError('Species {} has more than one different monomer compartment, '
                                     'which is not supported in SBML'.format(s))
            else:
                compartment_name = 'default'
            _check(sp.setCompartment(compartment_name))
            _check(sp.setName(str(s).replace('% ', '._br_')))
            _check(sp.setBoundaryCondition(i in fixed_species_idx))
            _check(sp.setConstant(False))
            _check(sp.setHasOnlySubstanceUnits(True))
            if i not in initial_species_idx:
                _check(sp.setInitialAmount(0.0))


        # Parameters

        for param in itertools.chain(self.model.parameters,
                                     self.model._derived_parameters):
            p = smodel.createParameter()
            _check(p)
            _check(p.setId(param.name))
            _check(p.setName(param.name))
            _check(p.setValue(param.value))
            _check(p.setConstant(True))

        # Reactions
        for i, reaction in enumerate(self.model.reactions_bidirectional):
            rxn = smodel.createReaction()
            _check(rxn)
            _check(rxn.setId('r{}'.format(i)))
            _check(rxn.setName(' + '.join(reaction['rule'])))
            _check(rxn.setReversible(reaction['reversible']))

            for sp in reaction['reactants']:
                reac = rxn.createReactant()
                _check(reac)
                _check(reac.setSpecies('__s{}'.format(sp)))
                _check(reac.setConstant(True))

            for sp in reaction['products']:
                prd = rxn.createProduct()
                _check(prd)
                _check(prd.setSpecies('__s{}'.format(sp)))
                _check(prd.setConstant(True))

            for symbol in reaction['rate'].free_symbols:
                if isinstance(symbol, pysb.Expression):
                    expr = symbol.expand_expr(expand_observables=True)
                    for sym in expr.free_symbols:
                        if not isinstance(sym, (pysb.Parameter, pysb.Expression)):
                            # Species reference, needs to be specified as modifier
                            modifier = rxn.createModifier()
                            _check(modifier)
                            _check(modifier.setSpecies(str(sym)))

            rate = rxn.createKineticLaw()
            _check(rate)
            rate_mathml = self._sympy_to_sbmlast(reaction['rate'])
            _check(rate.setMath(rate_mathml))

        # Observables
        for i, observable in enumerate(self.model.observables):
            # create an observable "parameter"
            obs = smodel.createParameter()
            _check(obs)
            _check(obs.setId('__obs{}'.format(i)))
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
            prop.addOption('strict', False)
            prop.addOption('setLevelAndVersion', True)
            prop.addOption('ignorePackages', True)
            _check(doc.convert(prop))

        return doc

    def export(self, level=(3, 2)):
        """
        Export the SBML for the PySB model associated with the exporter

        Requires libsbml package.

        Parameters
        ----------
        level: (int, int)
            The SBML level and version to use. The default is SBML level 3, version 2. Conversion
            to other levels/versions may not be possible or may lose fidelity.

        Returns
        -------
        string
            String containing the SBML output.
        """
        return libsbml.writeSBMLToString(self.convert(level=level))
