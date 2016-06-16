from __future__ import print_function
import re
import numpy
from pysb.integrate import odesolve
import matplotlib.cm as cm
import matplotlib.colors as colors
import sympy
import pandas
import seaborn as sns
import networkx as nx
import time as tm
from helper_functions import parse_name
from py2cytoscape.data.util_network import NetworkUtil as util
from collections import OrderedDict
from py2cytoscape.data.cyrest_client import CyRestClient


class OrderedGraph(nx.DiGraph):
    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict

cy = CyRestClient()
cy.session.delete()

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return numpy.ma.masked_array(numpy.interp(value, x, y))


def f2hex_edges(fx, vmin=-0.99, vmax=0.99):
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    f2rgb = cm.ScalarMappable(norm=norm, cmap=sns.diverging_palette(240, 10, as_cmap=True))
    rgb = [f2rgb.to_rgba(rate)[:3] for rate in fx]
    colors_hex = [0] * (len(rgb))
    for i, color in enumerate(rgb):
        colors_hex[i] = '#%02x%02x%02x' % tuple([255 * fc for fc in color])
    return colors_hex


def f2hex_nodes(fx, vmin, vmax, midpoint):
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=midpoint)
    f2rgb = cm.ScalarMappable(norm=norm, cmap=sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True))
    rgb = [f2rgb.to_rgba(rate)[:3] for rate in fx]
    colors_hex = [0] * (len(rgb))
    for i, color in enumerate(rgb):
        colors_hex[i] = '#%02x%02x%02x' % tuple([255 * fc for fc in color])
    return colors_hex


def r_link(graph, s, r, **attrs):
    nodes = ('s%d' % s, 's%d' % r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = reversed(nodes)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)


def change_node_colors(node, color, graph):
    n = graph.get_node(node)
    n.attr['fillcolor'] = color
    return


def change_edge_colors(edge, color, graph):
    n = graph.get_edge(*edge)
    n.attr['color'] = color
    return


def change_edge_size(edge, size, graph):
    n = graph.get_edge(*edge)
    n.attr['penwidth'] = size
    return


def magnitude(x):
    return numpy.floor(numpy.log10(x))


def range_normalization(x, min_x, max_x, a=0.1, b=15):
    return a + (x - min_x) * (b - a) / (max_x - min_x)


def sig_apop(t, f, td, ts):
    """Return the amount of substrate cleaved at time t.

    Keyword arguments:
    t -- time
    f -- is the fraction cleaved at the end of the reaction
    td -- is the delay period between TRAIL addition and half-maximal substrate cleavage
    ts -- is the switching time between initial and complete effector substrate  cleavage
    """
    return f - f / (1 + numpy.exp((t - td) / (4 * ts)))


class FluxVisualization:
    mach_eps = numpy.finfo(float).eps

    def __init__(self, model):
        self.model = model
        self.tspan = None
        self.y = None
        self.param_dict = None
        self.sp_graph = None
        self.colors_time_edges = None
        self.colors_time_nodes = None
        self.size_time_edges = None
        self.view1 = None
        self.node_name2id = None
        self.edge_name2id = None

    def visualize(self, tspan=None, param_values=None, verbose=False):
        if verbose:
            print("Solving Simulation")

        if tspan is not None:
            self.tspan = tspan
        elif self.tspan is None:
            raise Exception("'tspan' must be defined.")

        if param_values is not None:
            # accept vector of parameter values as an argument
            if len(param_values) != len(self.model.parameters):
                raise Exception("param_values must be the same length as model.parameters")
            if not isinstance(param_values, numpy.ndarray):
                param_values = numpy.array(param_values)
        else:
            # create parameter vector from the values in the model
            param_values = numpy.array([p.value for p in self.model.parameters])

        self.param_dict = dict((p.name, param_values[i]) for i, p in enumerate(self.model.parameters))

        # self.param_dict['L_0'] = 0
        #
        # y_pre = odesolve(self.model, self.tspan[:200], self.param_dict, verbose=verbose)
        #
        # self.tspan = self.tspan[201:]
        #
        # y_pre[-1][0] = 3000
        # # y_pre[-1][9] = 100
        #
        # y0 = [y_pre['__s%d' % i][-1] for i in range(len(self.model.species))]
        #
        # self.y = odesolve(self.model, self.tspan, param_values=self.param_dict, y0=y0, verbose=verbose)

        self.y = odesolve(self.model, self.tspan, param_values=self.param_dict)

        if verbose:
            print("Creating graph")
        self.species_graph()
        self.networkx2cytoscape_setup(self.sp_graph)

        # self.sp_graph.add_node('t',
        #                        label='time',
        #                        shape='oval',
        #                        fillcolor='white', style="filled", color="transparent",
        #                        fontsize="50",
        #                        margin="0,0",
        #                        pos="20,20!")

        self.edges_colors(self.y)
        self.nodes_colors(self.y)

        if verbose:
            "Updating network"
        for kx, time in enumerate(self.tspan):
            edges_size = self.size_time_edges.iloc[:, kx].to_dict()
            edges_color = self.colors_time_edges.iloc[:, kx].to_dict()
            self.update_network(edge_color=edges_color, edge_size=edges_size)
            tm.sleep(0.3)

    def edges_colors(self, y):
        all_rate_colors = {}
        all_rate_sizes = {}
        rxns_matrix = numpy.zeros((len(self.model.reactions_bidirectional), len(self.tspan)))
        for idx, reac in enumerate(self.model.reactions_bidirectional):
            rate_reac = reac['rate']
            for p in self.param_dict:
                rate_reac = rate_reac.subs(p, self.param_dict[p])
            variables = [atom for atom in rate_reac.atoms(sympy.Symbol) if not re.match(r'\d', str(atom))]
            func = sympy.lambdify(variables, rate_reac, modules=dict(sqrt=numpy.lib.scimath.sqrt))
            args = [y[str(l)] for l in variables]  # arguments to put in the lambdify function
            react_rate = func(*args)
            rxns_matrix[idx] = react_rate

        max_all_times = [max(rxns_matrix[:, col]) for col in range(numpy.shape(rxns_matrix)[1])]
        min_all_times = [min(rxns_matrix[:, col]) for col in range(numpy.shape(rxns_matrix)[1])]

        max_abs_flux = numpy.array([max(i, abs(j)) for i, j in zip(max_all_times, min_all_times)])
        max_flux_diff_area = max_abs_flux / numpy.power(numpy.array([10] * len(max_abs_flux)), magnitude(max_abs_flux))

        global_min_rate = -6
        global_max_rate = magnitude(numpy.abs(max(max_all_times)))

        for rxn in self.model.reactions_bidirectional:
            rate = rxn['rate']
            for p in self.param_dict:
                rate = rate.subs(p, self.param_dict[p])
            variables = [atom for atom in rate.atoms(sympy.Symbol) if not re.match(r'\d', str(atom))]
            func = sympy.lambdify(variables, rate, modules=dict(sqrt=numpy.lib.scimath.sqrt))
            args = [y[str(l)] for l in variables]  # arguments to put in the lambdify function
            react_rate = func(*args) + self.mach_eps
            react_rate_magnitudes = magnitude(numpy.abs(react_rate))
            react_rate_diff_area = react_rate / numpy.power(numpy.array([10] * len(react_rate)), react_rate_magnitudes)
            rate_colors = f2hex_edges(react_rate_diff_area)
            rate_sizes = range_normalization(react_rate_magnitudes, min_x=global_min_rate, max_x=global_max_rate)
            for rctan in rxn['reactants']:
                for pro in rxn['products']:
                    edges_id = self.edge_name2id['s' + str(rctan) + ',s' + str(pro)]
                    all_rate_colors[edges_id] = rate_colors
                    all_rate_sizes[edges_id] = rate_sizes
        all_colors = pandas.DataFrame(all_rate_colors).transpose()
        all_sizes = pandas.DataFrame(all_rate_sizes).transpose()

        self.size_time_edges = all_sizes
        self.colors_time_edges = all_colors
        return

    def nodes_colors(self, y):
        all_rate_colors = {}
        initial_conditions_values = [ic[1].value for ic in self.model.initial_conditions]
        # cparp_info = curve_fit(sig_apop, self.tspan, y['cPARP'], p0=[100, 100, 100])[0]
        # midpoint = sig_apop(cparp_info[1], cparp_info[0], cparp_info[1], cparp_info[2])
        max_ic = max(initial_conditions_values)

        for idx in range(len(self.model.species)):
            node_colors = f2hex_nodes(y['__s%d' % idx], vmin=0, vmax=max_ic, midpoint=max_ic / 2)
            all_rate_colors[self.node_name2id['s%d' % idx]] = node_colors
        all_nodes_colors = pandas.DataFrame(all_rate_colors).transpose()
        self.colors_time_nodes = all_nodes_colors
        return

    def species_graph(self):

        graph = OrderedGraph()
        for idx, cp in enumerate(self.model.species):
            species_node = 's%d' % idx

            graph.add_node(species_node,
                           {'label': parse_name(self.model.species[idx]),
                            'background-color': "#ccffcc",
                            'font-size': 35})

        for reaction in self.model.reactions_bidirectional:
            reactants = set(reaction['reactants'])
            products = set(reaction['products'])
            attr_reversible = {'source-arrow-shape': 'DIAMOND', 'target-arrow-shape': 'DELTA',
                               'source-arrow-fill': 'hollow', 'width': 3} if reaction['reversible'] else {
                'source-arrow-shape': 'NONE', 'width': 3, 'target-arrow-shape': 'DELTA'}
            for s in reactants:
                for p in products:
                    r_link(graph, s, p, **attr_reversible)
        self.sp_graph = graph
        return self.sp_graph

    def networkx2cytoscape_setup(self, g):
        pos = nx.drawing.nx_agraph.pygraphviz_layout(g, prog='dot', args="-Grankdir=LR")
        g_cy = cy.network.create_from_networkx(g)
        view_id_list = g_cy.get_views()
        self.view1 = g_cy.get_view(view_id_list[0], format='view')

        # Switch current visual style to a simple one...
        minimal_style = cy.style.create('Minimal')
        cy.style.apply(style=minimal_style, network=g_cy)

        self.node_name2id = util.name2suid(g_cy, 'node')
        self.edge_name2id = util.name2suid(g_cy, 'edge')

        node_x_values = {self.node_name2id[i]: pos[i][0] for i in pos}
        node_y_values = {self.node_name2id[i]: pos[i][1] for i in pos}

        node_label_values = {self.node_name2id[i[0]]: i[1]['label'] for i in g.nodes(data=True)}
        node_color_values = {self.node_name2id[i[0]]: i[1]['background-color'] for i in g.nodes(data=True)}
        edge_source_arrow_head = {self.edge_name2id[str(i[0]) + ',' + str(i[1])]: i[2]['source-arrow-shape'] for i in
                                  g.edges(data=True)}
        edge_target_arrow_head = {self.edge_name2id[str(i[0]) + ',' + str(i[1])]: i[2]['target-arrow-shape'] for i in
                                  g.edges(data=True)}

        self.view1.update_node_views(visual_property='NODE_X_LOCATION', values=node_x_values)
        self.view1.update_node_views(visual_property='NODE_Y_LOCATION', values=node_y_values)

        self.view1.update_node_views(visual_property='NODE_LABEL', values=node_label_values)
        self.view1.update_node_views(visual_property='NODE_FILL_COLOR', values=node_color_values)
        self.view1.update_edge_views(visual_property='EDGE_SOURCE_ARROW_SHAPE', values=edge_source_arrow_head)
        self.view1.update_edge_views(visual_property='EDGE_TARGET_ARROW_SHAPE', values=edge_target_arrow_head)
        return

    def update_network(self, edge_color, edge_size):
        self.view1.update_edge_views(visual_property='EDGE_WIDTH', values=edge_size)
        self.view1.update_edge_views(visual_property='EDGE_STROKE_UNSELECTED_PAINT', values=edge_color)
        self.view1.update_edge_views(visual_property='EDGE_SOURCE_ARROW_UNSELECTED_PAINT', values=edge_color)
        self.view1.update_edge_views(visual_property='EDGE_TARGET_ARROW_UNSELECTED_PAINT', values=edge_color)


def run_flux_visualization(model, tspan, parameters=None, verbose=False):
    fv = FluxVisualization(model)
    fv.visualize(tspan, parameters, verbose)
