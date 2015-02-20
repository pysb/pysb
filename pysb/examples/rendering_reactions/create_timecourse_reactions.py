from pysb.integrate import odesolve
from pysb.examples.tyson_oscillator import model
#from pysb.examples.robertson import model
#from earm.lopez_embedded import model
import numpy as np
import pygraphviz
import pysb
import re
import colorsys
import matplotlib.pyplot as plt
from pysb.tools.render_reactions import r_link
import networkx as nx

t = np.linspace(0,100,2000)
x = odesolve(model, t)
strides = 10
#TODO we remove the first 10 steps saying its equilibration. Need a better way!
y = np.zeros(((len(t)-10)/strides,len(model.species)))
for i in range(len(model.species)):
    # normalize the concentrations of each species from 0 to 1
    tmp = x['__s'+str(i)]
    tmp = tmp[10::strides]
    max  = np.max(tmp)
    min = np.min(tmp)
    if max - min == 0:
        y[:,i] = 1.
    else:
        tmp = (tmp-min )/ (max - min)
        y[:,i] = tmp
    #plt.title('s'+str(i))
    #plt.plot(t[10::strides],y[:,i])
    #plt.savefig('s'+str(i)+'.png')
    #plt.clf()

# Red and green scale rgb to hex
def rgb_to_hex(n):
    n = int(n *100)
    R = (255 * n) / 100
    G = (255 * (100 - n)) / 100
    B =  0
    return '#%02x%02x%02x' % (R,G,B)


def run(model,colors,savename):

    pysb.bng.generate_equations(model)

    graph = pygraphviz.AGraph(directed=True, rankdir="LR")
    ic_species = [cp for cp, parameter in model.initial_conditions]
    for i, cp in enumerate(model.species):
        species_node = 's%d' % i
        slabel = re.sub(r'% ', r'%\\l', str(cp))
        slabel += '\\l'
        color = rgb_to_hex(colors[i])
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
    # multiple programs exist. dot, circo, neato
    graph.draw( savename,prog="circo") 
    print 'saved %s' % savename
    return graph
time = t[10::strides]
for i in range(np.shape(y)[0]):
    graph = run(model,y[i,:],'example%05d.png' % i)
    for j in range(len(model.species)):
    #    plt.title('s'+str(j))
        plt.plot(time,y[:,j])
        plt.plot(time[i],y[i,j],'o')
    plt.savefig('species_all_%05d.png' % (i))
    plt.clf()



G = nx.from_agraph(graph)
#nx.draw(G)
#plt.draw()
#plt.show()
nx.write_gml(G,'example.gml')
import os
video_file_name = 'tyson_movie.avi'
if os.path.isfile(video_file_name):
    print "removing old video file. Check to see if this is correct"
    os.remove(video_file_name)
os.system('avconv -r 10 -i example%s.png %s' % ('%05d',video_file_name))
os.system('avconv -r 10 -i species_all_%s.png %s' % ('%05d','sall.avi'))
#os.system('avconv -r 10 -i species_2_%s.png %s' % ('%05d','s2.avi'))
#os.system('avconv -r 10 -i species_3_%s.png %s' % ('%05d','s3.avi'))
#os.system('avconv -r 10 -i species_4_%s.png %s' % ('%05d','s4.avi'))
#os.system('avconv -r 10 -i species_5_%s.png %s' % ('%05d','s5.avi'))
os.system('rm example*.png')
os.system('rm species*.png')
