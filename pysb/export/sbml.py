"""
Module containing a class for exporting a PySB model to SBML using libSBML

For information on how to use the model exporters, see the documentation
for :py:mod:`pysb.export`.
"""
import pysb
import pysb.bng
from pysb.export import Exporter
from sympy.printing.mathml import MathMLPrinter
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


def print_mathml(expr, **settings):
    return MathMLContentPrinter(settings).doprint(expr)


def _check(value):
    """If 'value' is None, prints an error message constructed using
    'message' and then exits with status code 1.  If 'value' is an integer,
    it assumes it is a libSBML return status code.  If the code value is
    LIBSBML_OPERATION_SUCCESS, returns without further action; if it is not,
    prints an error message constructed using 'message' along with text from
    libSBML explaining the meaning of the code, and exits with status code 1.
    """
    if value is None:
        raise ValueError('LibSBML returned a null value')
    elif type(value) is int:
        if value == libsbml.LIBSBML_OPERATION_SUCCESS:
            return
        else:
            err_msg = 'Error encountered converting to SBML. ' \
                      + 'LibSBML returned error code ' + str(value) + ': "' \
                      + libsbml.OperationReturnValue_toString(value).strip() + '"'
            raise ValueError(err_msg)
    else:
        return


def _mathml_expr_call(expr, as_ast=False):
    mathstr = '<apply><ci>{}</ci>'.format(expr.name)
    for sym in expr.expand_expr(expand_observables=True).free_symbols:
        if isinstance(sym, pysb.Expression):
            continue
        mathstr += '<ci>{}</ci>'.format(sym.name if isinstance(sym, pysb.Parameter) else sym)
    mathstr += '</apply>'
    if not as_ast:
        return mathstr
    mathstr = "<?xml version='1.0' encoding='UTF-8'?>" \
          "<math xmlns='http://www.w3.org/1998/Math/MathML'>{}</math>".format(mathstr)
    rate_mathml = libsbml.readMathMLFromString(mathstr)
    _check(rate_mathml)
    return rate_mathml


class SbmlExporter(Exporter):
    """A class for returning the SBML for a given PySB model.

    Inherits from :py:class:`pysb.export.Exporter`, which implements
    basic functionality for all exporters.
    """
    def __init__(self, *args, **kwargs):
        super(SbmlExporter, self).__init__(*args, **kwargs)
        if not libsbml:
            raise ImportError('The SbmlExporter requires the libsbml python package')

    def _sympy_to_sbmlast(self, sympy_expr, wrap_lambda=False):
        """
        Convert a sympy expression to the AST format used by libsbml

        wrap_lambda indicates whether the expression should be converted into a mathml lambda function
        """
        mathml = print_mathml(sympy_expr)

        if wrap_lambda:
            # For function definitions
            args = ''
            for sym in sympy_expr.free_symbols:
                if isinstance(sym, pysb.Expression):
                    continue
                args += '<bvar><ci> {} </ci></bvar>'.format(sym.name if isinstance(sym, pysb.Parameter) else sym)
            mathml = '<lambda>{}{}</lambda>'.format(args, mathml)
        else:
            # For rates expressions with <apply> wrapper, but not within the function definitions block
            for expr in self.model.expressions:
                mathml = mathml.replace('<ci>{}</ci>'.format(expr.name), _mathml_expr_call(expr))

        mathml = "<?xml version='1.0' encoding='UTF-8'?>" \
                 "<math xmlns='http://www.w3.org/1998/Math/MathML'>{}</math>".format(
            mathml,
        )
        rate_mathml = libsbml.readMathMLFromString(mathml)
        _check(rate_mathml)
        return rate_mathml

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
                _check(c.setSize(cpt.size.value))
                _check(c.setConstant(True))
        else:
            c = smodel.createCompartment()
            _check(c)
            _check(c.setId('default'))
            _check(c.setSpatialDimensions(3))
            _check(c.setSize(1))
            _check(c.setConstant(True))

        # Expressions
        for expr in self.model.expressions:
            f = smodel.createFunctionDefinition()
            _check(f)
            _check(f.setId(expr.name))
            _check(f.setMath(self._sympy_to_sbmlast(expr.expand_expr(expand_observables=True), wrap_lambda=True)))

        # Initial values/assignments
        initial_concs = [0.0] * len(self.model.species)
        for cp, param in self.model.initial_conditions:
            sp_idx = self.model.get_species_index(cp)
            if isinstance(param, pysb.Expression):
                ia = smodel.createInitialAssignment()
                _check(ia)
                _check(ia.setSymbol('__s{}'.format(sp_idx)))
                _check(ia.setMath(_mathml_expr_call(param, as_ast=True)))
                initial_concs[sp_idx] = None
            else:
                initial_concs[sp_idx] = param.value

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
            _check(sp.setBoundaryCondition(False))
            _check(sp.setConstant(False))
            _check(sp.setHasOnlySubstanceUnits(True))
            if initial_concs[i] is not None:
                _check(sp.setInitialAmount(initial_concs[i]))

        # Parameters
        for i, param in enumerate(self.model.parameters):
            if param in self.model.parameters_initial_conditions():
                continue
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
            _check(rxn.setName('r{}'.format(i)))
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
