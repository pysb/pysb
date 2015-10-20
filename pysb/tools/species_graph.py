#!/usr/bin/env python
"""
Usage
=====

Usage: ``python -m pysb.tools.species_graph mymodel.py > mymodel.dot``

Generates species(nodes) and edges between them by a model into the "dot" graph format which can
be visualized with Graphviz.

To create a PDF from the .dot file, use the "dot" command from Graphviz::

    dot mymodel.dot -T pdf -O

This will create mymodel.dot.pdf. You can also change the "dot" command to one
of the other Graphviz drawing tools for a different type of layout. Note that
you can pipe the output of render_reactions straight into Graphviz without
creating an intermediate .dot file, which is especially helpful if you are
making continuous changes to the model and need to visualize your changes
repeatedly::

    python -m pysb.tools.species_graph mymodel.py | dot -T pdf -o mymodel.pdf

Note that some PDF viewers will auto-reload a changed PDF, so you may not even
need to manually reopen it every time you rerun the tool.


"""

from __future__ import print_function
import pysb
import pysb.bng
import sympy
import re
import sys
import os
import pygraphviz

def run(model):
    """
    Render the reactions produced by a model into the "dot" graph format.

    Parameters
    ----------
    model : pysb.core.Model
        The model to render.

    Returns
    -------
    string
        The dot format output.
    """

    pysb.bng.generate_equations(model)

    graph = pygraphviz.AGraph(directed=True, rankdir="LR")
    ic_species = [cp for cp, parameter in model.initial_conditions]
    for i, cp in enumerate(model.species):
        species_node = 's%d' % i
        slabel = re.sub(r'% ', r'%\\l', str(cp))
        slabel += '\\l'
        color = "#ccffcc"
        # color species with an initial condition differently
        if len([s for s in ic_species if s.is_equivalent_to(cp)]):
            color = "#aaffff"
        graph.add_node(species_node,
                       label=species_node,
                       shape="Mrecord",
                       fillcolor=color, style="filled", color="transparent",
                       fontsize="12",
                       margin="0.06,0")
    for i, reaction in enumerate(model.reactions):       
        reactants = set(reaction['reactants'])
        products = set(reaction['products'])
        attr_reversible = {}
        for s in reactants:
            for p in products:
                r_link(graph, s, p, **attr_reversible)
                
    return graph.string()

def r_link(graph, s, r, **attrs):
    nodes = ('s%d' % s, 's%d' % r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = reversed(nodes)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)


usage = __doc__
usage = usage[1:]  # strip leading newline

if __name__ == '__main__':
    # sanity checks on filename
    if len(sys.argv) <= 1:
        print(usage, end="")
        exit()
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
        print("Error in model script:\n")
        raise
    # grab the 'model' variable from the module
    try:
        model = model_module.__dict__['model']
    except KeyError:
        raise Exception("File '%s' isn't a model file" % model_filename)
    print(run(model))


