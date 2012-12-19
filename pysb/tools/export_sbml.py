#!/usr/bin/env python

# FIXME this should use libsbml if available

import pysb
import pysb.bng
import sympy
from sympy.printing.mathml import MathMLPrinter
import re
import sys
import os
import itertools
import textwrap
from StringIO import StringIO

def indent(text, n=0):
    """Re-indent a multi-line string, strip leading newlines and trailing spaces"""
    text = text.lstrip('\n')
    text = textwrap.dedent(text)
    lines = text.split('\n')
    lines = [' '*n + l for l in lines]
    text = '\n'.join(lines)
    text = text.rstrip(' ')
    return text

class MathMLContentPrinter(MathMLPrinter):
    """Prints an expression to MathML without presentation markup"""
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

def run(model):
    output = StringIO()
    pysb.bng.generate_equations(model)

    output.write(
        """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2" level="2" version="1">
    <model>
        <listOfCompartments>
            <compartment id="default" name="default" spatialDimensions="3" size="1"/>
        </listOfCompartments>
""")

    ics = [[s, 0] for s in model.species]  # complexpattern, initial value
    for cp, ic_param in model.initial_conditions:
        ics[model.get_species_index(cp)][1] = ic_param.value
    output.write("        <listOfSpecies>\n")
    for i, (cp, value) in enumerate(ics):
        id = 's%d' % i
        metaid = 'metaid_%s' % id
        name = str(cp).replace('% ', '._br_')  # CellDesigner does something weird with % in names
        output.write('            <species id="%s" metaid="%s" name="%s" compartment="default" initialAmount="%.17g">\n'
                     % (id, metaid, name, value));
        output.write(get_species_annotation(metaid, cp))
        output.write('            </species>\n')
    output.write("        </listOfSpecies>\n")

    output.write("        <listOfParameters>\n")
    for i, param in enumerate(model.parameters_rules()):
        output.write('            <parameter id="%s" metaid="metaid_%s" name="%s" value="%.17g"/>\n'
                     % (param.name, param.name, param.name, param.value));
    output.write("        </listOfParameters>\n")

    output.write("        <listOfReactions>\n")
    for i, reaction in enumerate(model.reactions_bidirectional):
        reversible = str(reaction['reversible']).lower()
        output.write('            <reaction id="r%d" metaid="metaid_r%d" name="r%d" reversible="%s">\n'
                     % (i, i, i, reversible));
        output.write('                <listOfReactants>\n');
        for species in reaction['reactants']:
            output.write('                    <speciesReference species="s%d"/>\n' % species)
        output.write('                </listOfReactants>\n');
        output.write('                <listOfProducts>\n');
        for species in reaction['products']:
            output.write('                    <speciesReference species="s%d"/>\n' % species)
        output.write('                </listOfProducts>\n');
        mathml = '<math xmlns="http://www.w3.org/1998/Math/MathML">' \
            + print_mathml(reaction['rate']) + '</math>'
        output.write('                <kineticLaw>' + mathml + '</kineticLaw>\n');
        output.write('            </reaction>\n');
    output.write("        </listOfReactions>\n")

    output.write("    </model>\n</sbml>\n")
    return output.getvalue()

if __name__ == '__main__':
    # sanity checks on filename
    if len(sys.argv) <= 1:
        raise Exception("You must specify the filename of a model script")
    model_filename = sys.argv[1]
    if not os.path.exists(model_filename):
        raise Exception("File '%s' doesn't exist" % model_filename)
    if not re.search(r'\.py$', model_filename):
        raise Exception("File '%s' is not a .py file" % model_filename)
    sys.path.insert(0, os.path.dirname(model_filename))
    model_name = re.sub(r'\.py$', '', os.path.basename(model_filename))
    # import it
    try:
        # FIXME if the model has the same name as some other "real" module
        # which we use, there will be trouble
        # (use the imp package and import as some safe name?)
        model_module = __import__(model_name)
    except StandardError as e:
        print "Error in model script:\n"
        raise
    # grab the 'model' variable from the module
    try:
        model = model_module.__dict__['model']
    except KeyError:
        raise Exception("File '%s' isn't a model file" % model_filename)
    print run(model)



