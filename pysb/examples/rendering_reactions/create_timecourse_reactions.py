from pysb.integrate import odesolve
from pysb.examples.tyson_oscillator import model
#from pysb.examples.robertson import model
#from earm.lopez_embedded import model
import numpy as np
import matplotlib.pyplot as plt
from pysb.bng import run_ssa
import pygraphviz
import pysb
import re
import colorsys
import matplotlib.pyplot as plt
t = np.linspace(0,100,2000)
x = odesolve(model, t)
strides = 10
y = np.zeros(((len(t)-10)/strides,len(model.species)))
for i in range(len(model.species)):
    tmp = x['__s'+str(i)]
    tmp = tmp[10::strides]
    max  = np.max(tmp)
    min = np.min(tmp)
    if max - min == 0:
        y[:,i] = 1.
    else:
        tmp = (tmp-min )/ (max - min)
        print np.max(tmp),np.min(tmp)
        y[:,i] = tmp
    plt.title('s'+str(i))
    plt.plot(t[10::strides],y[:,i])
    plt.savefig('s'+str(i)+'.png')
    plt.clf()
print np.shape(y)
#quit()

def rgb_to_hex(n):
    n = int(n *100)
    R = (255 * n) / 100
    G = (255 * (100 - n)) / 100
    B =  0
    return '#%02x%02x%02x' % (R,G,B)

def r_link(graph, s, r, **attrs):
    nodes = ('s%d' % s, 's%d' % r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = reversed(nodes)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)

def run(model,colors,savename):

    pysb.bng.generate_equations(model)

    graph = pygraphviz.AGraph(directed=True, rankdir="LR")
    ic_species = [cp for cp, parameter in model.initial_conditions]
    for i, cp in enumerate(model.species):
        species_node = 's%d' % i
        slabel = re.sub(r'% ', r'%\\l', str(cp))
        slabel += '\\l'
        #color = "#ccffcc"
        #print cp,colors[i]
        color = rgb_to_hex(colors[i])
        # color species with an initial condition differently
        #if len([s for s in ic_species if s.is_equivalent_to(cp)]):
        #    color = "#aaffff"
        graph.add_node(species_node,
                       label=species_node,
                       shape="Mrecord",
                       fillcolor=color, style="filled", color="transparent",
                       fontsize="12",
                       margin="0.06,0")
        #graph.graph_attr.update(landscape='true',ranksep='0.1')
    for i, reaction in enumerate(model.reactions):       
        reactants = set(reaction['reactants'])
        products = set(reaction['products'])
        attr_reversible = {}
        for s in reactants:
            for p in products:
                r_link(graph, s, p, **attr_reversible)
    graph.draw( savename,prog="neato")
    print 'saved %s' % savename
    return graph
for i in range(np.shape(y)[0]):
    print y[i,:]
    graph = run(model,y[i,:],'example%05d.png' % i)

import networkx as nx

G = nx.from_agraph(graph)
#nx.draw(G)
#plt.draw()
#plt.show()
nx.write_gml(G,'example.gml')
import os
video_file_name = 'robertson_movie.avi'
if os.path.isfile(video_file_name):
    print "removing old video file. Check to see if this is correct"
    os.remove(video_file_name)
os.system('avconv -r 4 -i example%s.png %s' % ('%05d',video_file_name))
os.system('rm example*.png')
