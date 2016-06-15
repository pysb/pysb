import pysb.bng
import networkx as nx
from helper_functions import parse_name
from py2cytoscape.data.cyrest_client import CyRestClient
from py2cytoscape.data.util_network import NetworkUtil as util
from collections import OrderedDict


class OrderedGraph(nx.DiGraph):
    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict


cy = CyRestClient()
cy.session.delete()


def r_link_reactions(graph, s, r, **attrs):
    nodes = ('s%d' % s, 'r%d' % r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = reversed(nodes)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)


def r_link_species(graph, s, r, **attrs):
    nodes = ('s%d' % s, 's%d' % r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = reversed(nodes)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)


class RenderModel:
    def __init__(self, model):
        self.model = model
        self.graph = None

    def visualize(self, render_type='reactions', verbose=False):
        if verbose:
            "Creating network"
        if render_type == 'reactions':
            self.graph = self.render_reactions()
        elif render_type == 'species':
            self.graph = self.render_species()
        else:
            raise Exception("render_type must be defined between 'reactions' and 'species'")

        if verbose:
            "Passing network to Cytoscape"
        self.networkx2cytoscape(self.graph)

    def render_reactions(self):
        pysb.bng.generate_equations(self.model)

        graph = OrderedGraph()
        ic_species = [cp for cp, parameter in self.model.initial_conditions]
        for i, cp in enumerate(self.model.species):
            species_node = 's%d' % i
            color = "#ccffcc"
            # color species with an initial condition differently
            if len([s for s in ic_species if s.is_equivalent_to(cp)]):
                color = "#aaffff"
            graph.add_node(species_node, attr_dict=
            {'label': parse_name(self.model.species[i]),
             'font-size': "35",
             'shape': "roundrectangle",
             'background-color': color})
        for i, reaction in enumerate(self.model.reactions_bidirectional):
            reaction_node = 'r%d' % i
            graph.add_node(reaction_node, attr_dict=
            {'label': reaction_node,
             'font-size': "35",
             'shape': "roundrectangle",
             'background-color': "#D3D3D3"})
            reactants = set(reaction['reactants'])
            products = set(reaction['products'])
            modifiers = reactants & products
            reactants = reactants - modifiers
            products = products - modifiers
            attr_reversible = {'source-arrow-shape': 'DIAMOND', 'target-arrow-shape': 'ARROW',
                               'source-arrow-fill': 'hollow', 'width': 3} if reaction['reversible'] else {
                'source-arrow-shape': 'NONE', 'width': 6, 'target-arrow-shape': 'ARROW'}
            for s in reactants:
                r_link_reactions(graph, s, i, **attr_reversible)
            for s in products:
                r_link_reactions(graph, s, i, _flip=True, **attr_reversible)
            for s in modifiers:
                attr_arrow = {'source-arrow-shape': 'NONE', 'width': 6, 'target-arrow-shape': 'DELTA'}
                r_link_reactions(graph, s, i, **attr_arrow)

        self.graph = graph
        return self.graph

    def render_species(self):
        pysb.bng.generate_equations(self.model)
        graph = OrderedGraph()
        for idx, cp in enumerate(self.model.species):
            species_node = 's%d' % idx

            graph.add_node(species_node,
                           {'label': parse_name(self.model.species[idx]),
                            'font-size': "35",
                            'shape': "roundrectangle",
                            'background-color': "#ccffcc"})

        for reaction in self.model.reactions_bidirectional:
            reactants = set(reaction['reactants'])
            products = set(reaction['products'])
            attr_reversible = {'source-arrow-shape': 'DIAMOND', 'target-arrow-shape': 'ARROW',
                               'source-arrow-fill': 'hollow', 'width': 3} if reaction['reversible'] else {
                'source-arrow-shape': 'NONE', 'width': 6, 'target-arrow-shape': 'ARROW'}
            for s in reactants:
                for p in products:
                    r_link_species(graph, s, p, **attr_reversible)
        self.graph = graph
        return self.graph

    def networkx2cytoscape(self, g):
        pos = nx.drawing.nx_agraph.pygraphviz_layout(g, prog='dot', args="-Grankdir=LR")
        g_cy = cy.network.create_from_networkx(g)
        view_id_list = g_cy.get_views()
        view1 = g_cy.get_view(view_id_list[0], format='view')

        # Switch current visual style to a simple one...
        minimal_style = cy.style.create('Minimal')
        cy.style.apply(style=minimal_style, network=g_cy)

        node_name2id = util.name2suid(g_cy, 'node')
        edge_name2id = util.name2suid(g_cy, 'edge')

        node_x_values = {node_name2id[i]: pos[i][0] for i in pos}
        node_y_values = {node_name2id[i]: pos[i][1] for i in pos}

        for i in g.edges(data=True): print i[2]

        node_label_values = {node_name2id[i[0]]: i[1]['label'] for i in g.nodes(data=True)}
        node_color_values = {node_name2id[i[0]]: i[1]['background-color'] for i in g.nodes(data=True)}
        edge_source_arrow_head = {edge_name2id[str(i[0]) + ',' + str(i[1])]: i[2]['source-arrow-shape'] for i in
                                  g.edges(data=True)}
        edge_target_arrow_head = {edge_name2id[str(i[0]) + ',' + str(i[1])]: i[2]['target-arrow-shape'] for i in
                                  g.edges(data=True)}

        view1.update_node_views(visual_property='NODE_X_LOCATION', values=node_x_values)
        view1.update_node_views(visual_property='NODE_Y_LOCATION', values=node_y_values)

        view1.update_node_views(visual_property='NODE_LABEL', values=node_label_values)
        view1.update_node_views(visual_property='NODE_FILL_COLOR', values=node_color_values)
        view1.update_edge_views(visual_property='EDGE_SOURCE_ARROW_SHAPE', values=edge_source_arrow_head)
        view1.update_edge_views(visual_property='EDGE_TARGET_ARROW_SHAPE', values=edge_target_arrow_head)
        return


def run_render_model(model, render_type='reactions', verbose=False):
    rm = RenderModel(model)
    rm.visualize(render_type=render_type, verbose=verbose)
