#!/usr/bin/env python

# FIXME this should use libsbml if available

import pysb
import pysb.bng
import sympy
import re
import sys
import os
import pygraphviz

def run(model):
    pysb.bng.generate_equations(model)

    graph = pygraphviz.AGraph(rankdir="LR")
    for i, cp in enumerate(model.species):
        species_node = 's%d' % i
        graph.add_node(species_node,
                       #label=str(cp),
                       shape="Mrecord",
                       fillcolor="#ccffcc", style="filled", color="transparent",
                       fontsize="12",
                       margin="0.06,0")
    for i, reaction in enumerate(model.reactions):
        reaction_node = 'r%d' % i
        graph.add_node(reaction_node,
                       label=reaction_node,
                       shape="circle",
                       fillcolor="lightgray", style="filled", color="transparent",
                       fontsize="12",
                       width=".3", height=".3", margin="0.06,0")
        reactants = set(reaction['reactants'])
        products = set(reaction['products'])
        modifiers = reactants & products
        reactants = reactants - modifiers
        products = products - modifiers
        for s in reactants:
            r_link(graph, s, i)
        for s in products:
            r_link(graph, s, i, _flip=True)
        for s in modifiers:
            r_link(graph, s, i, style="dotted", arrowType="odot")
    return graph.string()

def r_link(graph, s, r, **kwargs):
    nodes = ('s%d' % s, 'r%d' % r)
    if kwargs.get('_flip'):
        nodes = reversed(nodes)
    graph.add_edge(*nodes, **kwargs)

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



