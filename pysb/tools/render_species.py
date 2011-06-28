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
        sgraph = graph.add_subgraph(name='cluster_s%d' % si, label='s%d' % si,
                                    color="gray75", fontsize="20")
        bonds = {}
        for mi, mp in enumerate(cp.monomer_patterns):
            mgraph = sgraph.add_subgraph(name=sgraph.name + '_%d' % mi, label=mp.monomer.name,
                                         fillcolor="gray90", style="filled",
                                         fontsize="12")
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
                mgraph.add_node(mgraph.name + '_%s' % site, label=site_label,
                                fillcolor="white", color="transparent", style="filled",
                                fontname="courier", fontsize=10,
                                width=0.2, height=0.2, fixedsize=True)
            for site, value in mp.site_conditions.items():
                site_bonds = []
                if isinstance(value, int):
                    site_bonds.append(value)
                elif isinstance(value, tuple):
                    site_bonds.append(value[1])
                elif isinstance(value, list):
                    site_bonds += value
                for b in site_bonds:
                    bonds.setdefault(b, []).append(mgraph.name + '_%s' % site)
        for bi, sites in bonds.items():
            sgraph.add_edge(sites, label=bi)
    return graph.string()

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
