"""
Module containing a class to return the StochKit XML equivalent of a model

Contains code based on the `gillespy <https://github.com/JohnAbel/gillespy>`
library with permission from author Brian Drawert.

For information on how to use the model exporters, see the documentation
for :py:mod:`pysb.export`.
"""
from pysb.export import Exporter, CompartmentsNotSupported
from pysb.core import as_complex_pattern, Expression, Parameter
from pysb.bng import generate_equations
import numpy as np
import sympy
import re
from collections import defaultdict
import itertools
try:
    import lxml.etree as etree
    pretty_print = True
except ImportError:
    import xml.etree.ElementTree as etree
    import xml.dom.minidom
    pretty_print = False


class StochKitExporter(Exporter):
    """A class for returning the Kappa for a given PySB model.

    Inherits from :py:class:`pysb.export.Exporter`, which implements
    basic functionality for all exporters.
    """
    @staticmethod
    def _species_to_element(species_num, species_val):
        e = etree.Element('Species')
        idElement = etree.Element('Id')
        idElement.text = species_num
        e.append(idElement)

        initialPopulationElement = etree.Element('InitialPopulation')
        initialPopulationElement.text = str(species_val)
        e.append(initialPopulationElement)

        return e


    @staticmethod
    def _parameter_to_element(param_name, param_val):
        e = etree.Element('Parameter')
        idElement = etree.Element('Id')
        idElement.text = param_name
        e.append(idElement)
        expressionElement = etree.Element('Expression')
        expressionElement.text = str(param_val)
        e.append(expressionElement)
        return e


    @staticmethod
    def _reaction_to_element(rxn_name, rxn_desc, propensity_fxn, reactants,
                             products):
        e = etree.Element('Reaction')

        idElement = etree.Element('Id')
        idElement.text = rxn_name
        e.append(idElement)

        descriptionElement = etree.Element('Description')
        descriptionElement.text = rxn_desc
        e.append(descriptionElement)

        typeElement = etree.Element('Type')
        if isinstance(propensity_fxn, (Parameter, float)):
            typeElement.text = 'mass-action'
            e.append(typeElement)
            rateElement = etree.Element('Rate')
            rateElement.text = propensity_fxn.name if isinstance(
                propensity_fxn, Parameter) else str(propensity_fxn)
            e.append(rateElement)
        else:
            typeElement.text = 'customized'
            e.append(typeElement)
            functionElement = etree.Element('PropensityFunction')
            functionElement.text = propensity_fxn
            e.append(functionElement)

        reactantElement = etree.Element('Reactants')

        for reactant, stoichiometry in reactants.items():
            srElement = etree.Element('SpeciesReference')
            srElement.set('id', reactant)
            srElement.set('stoichiometry', str(stoichiometry))
            reactantElement.append(srElement)

        e.append(reactantElement)

        productElement = etree.Element('Products')
        for product, stoichiometry in products.items():
            srElement = etree.Element('SpeciesReference')
            srElement.set('id', product)
            srElement.set('stoichiometry', str(stoichiometry))
            productElement.append(srElement)
        e.append(productElement)

        return e

    def export(self, initials=None, param_values=None):
        """Generate the corresponding StochKit2 XML for a PySB model

        Parameters
        ----------
        initials : list of numbers
            List of initial species concentrations overrides
            (must be same length as model.species). If None,
            the concentrations from the model are used.
        param_values : list
            List of parameter value overrides (must be same length as
            model.parameters). If None, the parameter values from the model
            are used.

        Returns
        -------
        string
            The model in StochKit2 XML format
        """
        if self.model.compartments:
            raise CompartmentsNotSupported()

        generate_equations(self.model)
        document = etree.Element("Model")

        d = etree.Element('Description')

        d.text = 'Exported from PySB Model: %s' % self.model.name
        document.append(d)

        # Number of Reactions
        nr = etree.Element('NumberOfReactions')
        nr.text = str(len(self.model.reactions))
        document.append(nr)

        # Number of Species
        ns = etree.Element('NumberOfSpecies')
        ns.text = str(len(self.model.species))
        document.append(ns)

        if param_values is None:
            # Get parameter values from model if not supplied
            param_values = [p.value for p in self.model.parameters]

        # Add in derived parameters if needed
        if self.model._derived_parameters and len(param_values) == len(
                self.model.parameters):
            param_values += [p.value for p in self.model._derived_parameters]
        elif len(param_values) != len(self.model.parameters) + len(
                self.model._derived_parameters
        ):
            raise ValueError('param_values must be a list of numeric '
                             'parameter values the same length as '
                             'model.parameters, optionally including '
                             'derived parameters on the end (if model contains '
                             'local functions)')

        # Get initial species concentrations from model if not supplied
        if initials is None:
            initials = np.zeros((len(self.model.species),))
            subs = dict((p, param_values[i]) for i, p in
                        enumerate(self.model.parameters))

            for ic in self.model.initials:
                cp = as_complex_pattern(ic.pattern)
                si = self.model.get_species_index(cp)
                if si is None:
                    raise IndexError("Species not found in model: %s" %
                                     repr(cp))
                if ic.value in self.model.parameters:
                    pi = self.model.parameters.index(ic.value)
                    value = param_values[pi]
                elif ic.value in self.model.expressions:
                    value = ic.value.expand_expr().evalf(subs=subs)
                else:
                    raise ValueError(
                        "Unexpected initial condition value type")
                initials[si] = value
        else:
            # Validate length
            if len(initials) != len(self.model.species):
                raise Exception('initials must be a list of numeric initial '
                                'concentrations the same length as '
                                'model.species')

        # Species
        spec = etree.Element('SpeciesList')
        for s_id in range(len(self.model.species)):
            spec.append(self._species_to_element('__s%d' % s_id,
                                                 initials[s_id]))
        document.append(spec)

        # Parameters
        params = etree.Element('ParametersList')
        for p_id, param in enumerate(itertools.chain(
                self.model.parameters, self.model._derived_parameters)):
            p_name = param.name
            if p_name == 'vol':
                p_name = '__vol'
            p_value = param.value if param_values is None else \
                param_values[p_id]
            params.append(self._parameter_to_element(p_name, p_value))
        # Default volume parameter value
        params.append(self._parameter_to_element('vol', 1.0))

        document.append(params)

        # Expressions and observables
        expr_strings = {
            e.name: '(%s)' % sympy.ccode(
                e.expand_expr(expand_observables=True)
            )
            for e in itertools.chain(
                self.model.expressions_constant(),
                self.model.expressions_dynamic(include_local=False),
                self.model._derived_expressions
            )
        }

        # Reactions
        reacs = etree.Element('ReactionsList')
        pattern = re.compile("(__s\d+)\*\*(\d+)")
        for rxn_id, rxn in enumerate(self.model.reactions):
            rxn_name = 'Rxn%d' % rxn_id
            rxn_desc = 'Rules: %s' % str(rxn["rule"])

            reactants = defaultdict(int)
            products = defaultdict(int)
            # reactants
            for r in rxn["reactants"]:
                reactants["__s%d" % r] += 1
            # products
            for p in rxn["products"]:
                products["__s%d" % p] += 1
            # replace terms like __s**2 with __s*(__s-1)
            rate = str(rxn["rate"])

            matches = pattern.findall(rate)
            for m in matches:
                repl = m[0]
                for i in range(1, int(m[1])):
                    repl += "*(%s-%d)" % (m[0], i)
                rate = re.sub(pattern, repl, rate, count=1)

            # expand only expressions used in the rate eqn
            for e in {sym for sym in rxn["rate"].atoms()
                      if isinstance(sym, Expression)}:
                rate = re.sub(r'\b%s\b' % e.name,
                              expr_strings[e.name],
                              rate)

            total_reactants = sum(reactants.values())
            rxn_params = rxn["rate"].atoms(Parameter)
            rate = None
            if total_reactants <= 2 and len(rxn_params) == 1:
                # Try to parse as mass action to avoid compiling custom
                # propensity functions in StochKit (slow for big models)
                rxn_param = rxn_params.pop()
                putative_rate = sympy.Mul(*[sympy.symbols(r) ** r_stoich for
                                            r, r_stoich in
                                            reactants.items()]) * rxn_param

                rxn_floats = rxn["rate"].atoms(sympy.Float)
                rate_mul = 1.0
                if len(rxn_floats) == 1:
                    rate_mul = next(iter(rxn_floats))
                    putative_rate *= rate_mul

                if putative_rate == rxn["rate"]:
                    # Reaction is mass-action, set rate to a Parameter or float
                    if len(rxn_floats) == 0:
                        rate = rxn_param
                    elif len(rxn_floats) == 1:
                        rate = rxn_param.value * float(rate_mul)

                    if rate is not None and len(reactants) == 1 and \
                            max(reactants.values()) == 2:
                        # Need rate * 2 in addition to any rate factor
                        rate = (rate.value if isinstance(rate, Parameter)
                                else rate) * 2.0

            if rate is None:
                # Custom propensity function needed

                if isinstance(rxn['rate'], Expression):
                    rate = expr_strings[rxn['rate'].name]
                else:
                    rxn_atoms = rxn["rate"].atoms()

                    # replace terms like __s**2 with __s*(__s-1)
                    rate = str(rxn["rate"])

                    matches = pattern.findall(rate)
                    for m in matches:
                        repl = m[0]
                        for i in range(1, int(m[1])):
                            repl += "*(%s-%d)" % (m[0], i)
                        rate = re.sub(pattern, repl, rate, count=1)

                    # expand only expressions used in the rate eqn
                    for e in {sym for sym in rxn_atoms
                              if isinstance(sym, Expression)}:
                        rate = re.sub(r'\b%s\b' % e.name,
                                      expr_strings[e.name],
                                      rate)

            reacs.append(self._reaction_to_element(rxn_name,
                                                   rxn_desc,
                                                   rate,
                                                   reactants,
                                                   products))
        document.append(reacs)

        if pretty_print:
            return etree.tostring(document, pretty_print=True).decode('utf8')
        else:
            # Hack to print pretty xml without pretty-print
            # (requires the lxml module).
            doc = etree.tostring(document)
            xmldoc = xml.dom.minidom.parseString(doc)
            uglyXml = xmldoc.toprettyxml(indent='  ')
            text_re = re.compile(">\n\s+([^<>\s].*?)\n\s+</", re.DOTALL)
            prettyXml = text_re.sub(">\g<1></", uglyXml)
            return prettyXml
