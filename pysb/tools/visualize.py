import subprocess
import os
import pysb
from pysb.tools.render_reactions import r_link
from pysb.bng import generate_equations , _get_bng_path
from pysb.generator.bng import BngGenerator
import pygraphviz
import re
import networkx as nx
from  networkx.utils.decorators import is_string_like
import numpy as np

def create_visualization_bngl(model):
    # creates all visualize types using bngl visualization
    
    commands = """ 
    #visualize({type=>'process'})
    visualize({type=>'contactmap'}) \n 
    visualize({type=>'ruleviz_pattern'}) \n 
    visualize({type=>'ruleviz_operation'}) \n
    visualize({type=>'regulatory','background'=>1,suffix=>1})
    visualize({type=>'regulatory','background'=>1,'groups' =>1, suffix=>2}) 
    visualize({type=>'regulatory','background'=>1,collapse=>1, suffix=>3})
    visualize({type=>'regulatory','background'=>1,'groups' =>1,collapse=>1, suffix=>4})
    """
    bng_filename = 'visualize%s.bngl' % model.name
    bng_file = open(bng_filename, 'w')
    bng_file.write(BngGenerator(model).get_content())
    bng_file.write(commands)
    bng_file.close()
    p = subprocess.Popen(['perl', _get_bng_path(), bng_filename],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in iter(p.stdout.readline, b''):
        print line,
    os.unlink(bng_filename)


def rgb_to_hex(n):
    # Red and green scale rgb to hex
    
    n = int(n *100)
    R = (255 * (100 - n)) / 100
    G = (255 * n) / 100
    B =  0
    return '#%02x%02x%02x' % (R,G,B)

def create_png_image_from_dot(model,colors,savename):

    pysb.bng.generate_equations(model)

    graph = pygraphviz.AGraph(directed=True,rankdir="LR")
    ic_species = [cp for cp, parameter in model.initial_conditions]
    for i, cp in enumerate(model.species):
        species_node = 's%d' % i
        slabel = re.sub(r'% ', r'%\\l', str(cp))
        slabel += '\\l'
        
        
#         if colors[i] < 10**(-12.):
#             outline_color = "black"
#             color = "yellow"
#         if colors[i] == 1.:
#             outline_color = "black"
#             color = "blue"
        #else:
        outline_color = "transparent"
        color = rgb_to_hex(colors[i])
        graph.add_node(species_node,
                       #label=species_node,
                       label = slabel,
                       shape="Mrecord",
                       fillcolor=color, style="filled", color=outline_color,
                       fontsize="12",
                       margin="0.06,0")
    for i, reaction in enumerate(model.reactions):       
        reactants = set(reaction['reactants'])
        products = set(reaction['products'])
        attr_reversible = {}
        for s in reactants:
            for p in products:
                r_link(graph, s, p, **attr_reversible)    
    print "saved image as %s" % savename
    #multiple programs exist to render graphs
    #dot filter for drawing directed graphs
    #neato filter for drawing undirected graphs
    #twopi filter for radial layouts of graphs
    #circo filter for circular layout of graphs
    #fdp  filter for drawing undirected graphs
    #sfdp filter for drawing large undirected graphs
    graph.draw( savename,prog="dot")

def generate_gml(G):

    def string_item(k,v,indent):
        if is_string_like(v):
            v='"%s"'%v
        elif type(v)==bool:
            v=int(v)
        else:
            return ''
        return "%s %s"%(k,v)

    # check for attributes or assign empty dict
    if hasattr(G,'graph_attr'):
        graph_attr=G.graph_attr
    else:
        graph_attr={}
    if hasattr(G,'node_attr'):
        node_attr=G.node_attr
    else:
        node_attr={}

    indent=2*' '
    count=iter(range(len(G)))
    node_id={}

    yield "graph ["
    if G.is_directed():
        yield indent+"directed 1"
    # write graph attributes 
    for k,v in G.graph.items():
        if k == 'directed':
            continue
        yield indent+string_item(k,v,indent)
    # write nodes
    for n in G:
        yield indent+"node ["
        # get id or assign number
        nid=G.node[n].get('id',next(count))
        node_id[n]=nid
        yield 2*indent+"id %s"%nid
        label=G.node[n].get('label',n)
        if is_string_like(label):
            label='"%s"'%label
        yield 2*indent+'label %s'%label
        if n in G:
          for k,v in G.node[n].items():
              if k=='id' or k == 'label': continue
              yield 2*indent+string_item(k,v,indent)
        yield indent+"]"
    # write edges
    for u,v,edgedata in G.edges_iter(data=True):
        yield indent+"edge ["
        yield 2*indent+"source %s"%node_id[u]
        yield 2*indent+"target %s"%node_id[v]
        for k,v in edgedata.items():
            if k=='source': continue
            if k=='target': continue
            yield 2*indent+string_item(k,v,indent)
        yield indent+"]"
    yield "]"


def write_gml(G, path):
    file = open(path,'w')
    output=''
    for line in generate_gml(G):
        print line
        output += line
        #path.write(line.encode('latin-1'))
        output += '\n'
    file.write(output)
    file.close()    