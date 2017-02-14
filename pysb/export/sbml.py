"""
Module containing a class for exporting a PySB model to SBML.

For information on how to use the model exporters, see the documentation
for :py:mod:`pysb.export`.
"""

# FIXME this should use libsbml if available

import pysb
import pysb.bng
from pysb.export import Exporter
import sympy
from sympy.printing.mathml import MathMLPrinter
import itertools
import textwrap
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

def indent(text, n=0):
    """Re-indent a multi-line string, stripping leading newlines and trailing
    spaces."""
    text = text.lstrip('\n')
    text = textwrap.dedent(text)
    lines = text.split('\n')
    lines = [' '*n + l for l in lines]
    text = '\n'.join(lines)
    text = text.rstrip(' ')
    return text

class MathMLContentPrinter(MathMLPrinter):
    """Prints an expression to MathML without presentation markup."""
    def _print_Symbol(self, sym):
        ci = self.dom.createElement(self.mathml_tag(sym))
        ci.appendChild(self.dom.createTextNode(sym.name))
        return ci

def print_mathml(expr, **settings):
    return MathMLContentPrinter(settings).doprint(expr)

def format_single_annotation(annotation):
    return '''<rdf:li rdf:resource="%s" />''' % annotation.object

def get_annotation_preamble(meta_id):
    return indent('''
      <annotation>
          <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:bqbiol="http://biomodels.net/biology-qualifiers/" xmlns:bqmodel="http://biomodels.net/model-qualifiers/">
              <rdf:Description rdf:about="#%s">
      ''' % meta_id)

def get_annotation_postamble():
    return indent('''
              </rdf:Description>
          </rdf:RDF>
      </annotation>
      ''')

def get_species_annotation(meta_id, cp):
    output = get_annotation_preamble(meta_id)
    model = cp.monomer_patterns[0].monomer.model()
    if len(cp.monomer_patterns) == 1:
        # single monomer
        all_annotations = model.get_annotations(cp.monomer_patterns[0].monomer)
        groups = itertools.groupby(all_annotations, lambda a: a.predicate)
        for predicate, annotations in groups:
            qualifier = 'bqbiol:%s' % predicate
            output += indent('<%s>\n    <rdf:Bag>\n' % qualifier, 12)
            for a in annotations:
                output += indent(format_single_annotation(a) + '\n', 20)
            output += indent('    </rdf:Bag>\n</%s>\n' % qualifier, 12)
    else:
        # complex
        monomers = set(mp.monomer for mp in cp.monomer_patterns)
        annotations = [a for m in monomers for a in model.get_annotations(m)
                       if a.predicate in ('is', 'hasPart')]
        qualifier = 'bqbiol:hasPart'
        output += indent('<%s>\n    <rdf:Bag>\n' % qualifier, 12)
        for a in annotations:
            output += indent(format_single_annotation(a) + '\n', 20)
        output += indent('    </rdf:Bag>\n</%s>\n' % qualifier, 12)
    output += get_annotation_postamble()
    return indent(output, 16)

class SbmlExporter(Exporter):
    """A class for returning the SBML for a given PySB model.

    Inherits from :py:class:`pysb.export.Exporter`, which implements
    basic functionality for all exporters.
    """

    def export(self):
        """Export the SBML for the PySB model associated with the exporter.

        Returns
        -------
        string
            String containing the SBML output.
        """

        output = StringIO()
        pysb.bng.generate_equations(self.model)

        output.write(
            """<?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level2" level="2" version="1">
        <model name="%s">""" % self.model.name)

        if self.docstring:
            notes_str = """
            <notes>
                <body xmlns="http://www.w3.org/1999/xhtml">
                    <p>%s</p>
                </body>
            </notes>""" % self.docstring.replace("\n", "<br />\n"+" "*20)
            output.write(notes_str)

        output.write("""
            <listOfCompartments>
                <compartment id="default" name="default" spatialDimensions="3" size="1"/>
            </listOfCompartments>
    """)

        # complexpattern, initial value
        ics = [[s, 0] for s in self.model.species]
        for cp, ic_param in self.model.initial_conditions:
            ics[self.model.get_species_index(cp)][1] = ic_param.value
        output.write("        <listOfSpecies>\n")
        for i, (cp, value) in enumerate(ics):
            id = '__s%d' % i
            metaid = 'metaid_%d' % i
            name = str(cp).replace('% ', '._br_')  # CellDesigner does something weird with % in names
            output.write('            <species id="%s" metaid="%s" name="%s" compartment="default" initialAmount="%.17g">\n'
                         % (id, metaid, name, value));
            output.write(get_species_annotation(metaid, cp))
            output.write('            </species>\n')
        output.write("        </listOfSpecies>\n")

        output.write("        <listOfParameters>\n")
        for i, param in enumerate(self.model.parameters_rules()):
            output.write('            <parameter id="%s" metaid="metaid_%s" name="%s" value="%.17g"/>\n'
                         % (param.name, param.name, param.name, param.value));
        output.write("        </listOfParameters>\n")

        output.write("        <listOfReactions>\n")
        for i, reaction in enumerate(self.model.reactions_bidirectional):
            reversible = str(reaction['reversible']).lower()
            output.write('            <reaction id="r%d" metaid="metaid_r%d" name="r%d" reversible="%s">\n'
                         % (i, i, i, reversible));
            output.write('                <listOfReactants>\n');
            for species in reaction['reactants']:
                output.write('                    <speciesReference species="__s%d"/>\n' % species)
            output.write('                </listOfReactants>\n');
            output.write('                <listOfProducts>\n');
            for species in reaction['products']:
                output.write('                    <speciesReference species="__s%d"/>\n' % species)
            output.write('                </listOfProducts>\n');
            mathml = '<math xmlns="http://www.w3.org/1998/Math/MathML">' \
                + print_mathml(reaction['rate']) + '</math>'
            output.write('                <kineticLaw>' + mathml + '</kineticLaw>\n');
            output.write('            </reaction>\n');
        output.write("        </listOfReactions>\n")

        output.write("    </model>\n</sbml>\n")
        return output.getvalue()


