#!/usr/bin/env python

import sys
import os
import re
import pygraphviz
import pysb.bng

def run(model):
    pysb.bng.generate_equations(model)
    graph = pygraphviz.AGraph(rankdir="LR")
    for si, cp in enumerate(model.species):
        sgraph_name = 'cluster_s%d' % si
        sgraph = graph.add_subgraph(sgraph_name, label='s%d' % si,
                                    color="gray75", fontsize="20")
        bonds = {}
        for mi, mp in enumerate(cp.monomer_patterns):
            monomer_node = '%s_%d' % (sgraph_name, mi)
            sgraph.add_node(monomer_node,
                            label=mp.monomer.name,
                            shape="rectangle",
                            fillcolor="lightblue", style="filled",
                            fontsize="12",
                            width=".3", height=".3", margin="0.06,0")
            for site in mp.monomer.sites:
                site_state = None
                cond = mp.site_conditions[site]
                if isinstance(cond, str):
                    site_state = cond
                elif isinstance(cond, tuple):
                    site_state = cond[0]
                site_label = site
                if site_state is not None:
                    site_label += '=%s' % site_state
                site_node = '%s_%s' % (monomer_node, site)
                sgraph.add_node(site_node, label=site_label,
                                fontname="courier", fontsize='10',
                                fillcolor="yellow", style="filled", color="transparent",
                                width=".2", height=".2", margin="0")
                sgraph.add_edge(monomer_node, site_node,
                                style="bold")
            for site, value in mp.site_conditions.items():
                site_bonds = []
                if isinstance(value, int):
                    site_bonds.append(value)
                elif isinstance(value, tuple):
                    site_bonds.append(value[1])
                elif isinstance(value, list):
                    site_bonds += value
                for b in site_bonds:
                    bonds.setdefault(b, []).append('%s_%s' % (monomer_node, site))
        for bi, sites in bonds.items():
            sgraph.add_edge(sites, label=str(bi),
                            style="dotted")
    return graph.string()


usage = """
Usage: python -m pysb.tools.render_species mymodel.py > mymodel.dot

Renders the species from a model into the "dot" graph format which can be
visualized with Graphviz.

To create a PDF from the .dot file, use the "neato" command from Graphviz:

    neato mymodel.dot -T pdf -O

This will create mymodel.dot.pdf. You can also try "dot" instead of "neato" for
a different type of layout. Alternately, the following "one-liner" may be
convenient if you are making continuous changes to the model and need to run the
tool repeatedly:

    python -m pysb.tools.render_species mymodel.py | neato -T pdf -o mymodel.pdf

Note that some PDF viewers will auto-reload a changed PDF, so you may not even
need to manually reopen it every time you rerun the tool.
"""
usage = usage[1:]  # strip leading newline

if __name__ == '__main__':
    # sanity checks on filename
    if len(sys.argv) <= 1:
        print usage,
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
        # FIXME if the model has the same name as some other "real" module which we use,
        # there will be trouble (use the imp package and import as some safe name?)
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
