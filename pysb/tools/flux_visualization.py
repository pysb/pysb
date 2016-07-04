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
import threading

# Cytoscape session
cy = CyRestClient()
cy.session.delete()


class OrderedGraph(nx.DiGraph):
    """
    Networkx Digraph that stores the nodes in the order they are input
    """
    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict


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
    """

    :param fx: Vector of reaction rates (flux)
    :param vmin: Value of minimum for normalization
    :param vmax: Value of maximum for normalization
    :return: This function returns a vector of colors in hex format that represents flux
    """
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    f2rgb = cm.ScalarMappable(norm=norm, cmap=sns.diverging_palette(240, 10, as_cmap=True))
    rgb = [f2rgb.to_rgba(rate)[:3] for rate in fx]
    colors_hex = [0] * (len(rgb))
    for i, color in enumerate(rgb):
        colors_hex[i] = '#%02x%02x%02x' % tuple([255 * fc for fc in color])
    return colors_hex


def f2hex_nodes(fx, vmin, vmax, midpoint):
    """

    :param fx: Vector of species concentration
    :param vmin: Value of minimum for normalization
    :param vmax: Value of maximum for normalization
    :param midpoint: Value of midpoint for normalization
    :return: Returns a vector of colors in hex format that represents species concentration
    """
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=midpoint)
    f2rgb = cm.ScalarMappable(norm=norm, cmap=sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True))
    rgb = [f2rgb.to_rgba(rate)[:3] for rate in fx]
    colors_hex = [0] * (len(rgb))
    for i, color in enumerate(rgb):
        colors_hex[i] = '#%02x%02x%02x' % tuple([255 * fc for fc in color])
    return colors_hex


def r_link(graph, s, r, **attrs):
    """

    :param graph: Networkx graph
    :param s: Source node
    :param r: Target node
    :param attrs: Other attributes for edge
    :return: Links two nodes through an edge
    """
    nodes = ('s%d' % s, 's%d' % r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = nodes[::-1]
    attrs.setdefault('arrowhead', 'normal')
    attrs['name'] = ','.join(i for i in nodes)
    graph.add_edge(*nodes, **attrs)


def change_node_colors(node, color, graph):
    """

    :param node: Node whose color is going to be changed
    :param color: Color assigned to the node
    :param graph: Networkx graph
    :return: Updates node color
    """
    n = graph.get_node(node)
    n.attr['fillcolor'] = color
    return


def change_edge_colors(edge, color, graph):
    """

    :param edge: Edge whose color is going to be changed
    :param color: Color assigned to the edge
    :param graph: Networkx graph
    :return: Updates edge color
    """
    n = graph.get_edge(*edge)
    n.attr['color'] = color
    return


def change_edge_size(edge, size, graph):
    """

    :param edge: Edge whose size is going to be changed
    :param size: Size assigned to the edge
    :param graph: Networkx graph
    :return: Updates edge size
    """
    n = graph.get_edge(*edge)
    n.attr['penwidth'] = size
    return


def magnitude(x):
    """

    :param x: Vector of numbers
    :return: Magnitude of numbers
    """
    return numpy.floor(numpy.log10(x))


def range_normalization(x, min_x, max_x, a=0.1, b=15):
    """

    :param x: Vector of numbers to be normalized
    :param min_x: Minimum value in vector x
    :param max_x: Maximum value in vector x
    :param a: Value of minimum used for the normalization
    :param b: Value of maximum used for the normalization
    :return: Normalized vector
    """
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


def button_state(state):
    return state


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
        """

        :param tspan: time span for the simulation
        :param param_values: Parameter values for the model
        :param verbose: Verbose option
        :return: Connects to cytoscape and visualize flow
        """
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

        # Parameters in a dictionary form
        self.param_dict = dict((p.name, param_values[i]) for i, p in enumerate(self.model.parameters))

        # Solution of the differential equations
        self.y = odesolve(self.model, self.tspan, param_values=self.param_dict)

        if verbose:
            print("Creating graph")

        # Creates the graph from the model
        self.species_graph()
        # self.sp_graph.add_node('t', {'label': 'time', 'background-color': '#fffaf0'})

        # Sets up the graph in cytoscape
        self.networkx2cytoscape_setup(self.sp_graph)

        # Creates panda data frame with color values for edges and nodes.
        self.edges_colors_sizes(self.y)
        self.nodes_colors(self.y)

        if verbose:
            "Updating network"

        self.restartable_for(self.tspan)

        # # TODO: the while loop consumes cpu as long as the stop nodes is selected, find a different way to do this
        # for kx, time in enumerate(self.tspan):
        #     stop_state = button_state(self.view1.get_node_views_as_dict()[self.node_name2id['Stop']]['NODE_SELECTED'])
        #     while stop_state:
        #         tm.sleep(1)
        #         play_state = button_state(
        #             self.view1.get_node_views_as_dict()[self.node_name2id['Play']]['NODE_SELECTED'])
        #         if play_state:
        #             break
        #
        #     edges_size = self.size_time_edges.iloc[:, kx].to_dict()
        #     edges_color = self.colors_time_edges.iloc[:, kx].to_dict()
        #     time_stamp = {self.node_name2id['t']: 'Time:' + ' ' + '%d' % time + ' ' + 'sec'}
        #     self.update_network(edge_color=edges_color, edge_size=edges_size, time_stamp=time_stamp)
        #     tm.sleep(0.5)

    def restartable_for(self, sequence):
        restart = False
        for kx, time in enumerate(sequence):
            stop_state = button_state(self.view1.get_node_views_as_dict()[self.node_name2id['Stop']]['NODE_SELECTED'])
            while stop_state:
                tm.sleep(1)
                play_state = button_state(
                    self.view1.get_node_views_as_dict()[self.node_name2id['Play']]['NODE_SELECTED'])
                if play_state:
                    break

            edges_size = self.size_time_edges.iloc[:, kx].to_dict()
            edges_color = self.colors_time_edges.iloc[:, kx].to_dict()
            time_stamp = {self.node_name2id['t']: 'Time:' + ' ' + '%d' % time + ' ' + 'sec'}
            self.update_network(edge_color=edges_color, edge_size=edges_size, time_stamp=time_stamp)
            tm.sleep(0.5)
            restart_state = button_state(
                self.view1.get_node_views_as_dict()[self.node_name2id['Restart']]['NODE_SELECTED'])
            if restart_state:
                self.view1.update_node_views(visual_property='NODE_SELECTED',
                                             values={self.node_name2id['Restart']: False})

                self.view1.update_node_views(visual_property='NODE_SELECTED',
                                             values={self.node_name2id['Play']: True})
                restart = True
                break
        if restart:
            self.restartable_for(sequence)

    def edges_colors_sizes(self, y):
        """

        :param y: Solution of the differential equations (from odesolve)
        :return: Returns a pandas data frame where each row is an edge and each column a time point, with the color value
        """
        all_rate_colors = {}
        all_rate_sizes = {}
        rxns_matrix = numpy.zeros((len(self.model.reactions_bidirectional), len(self.tspan)))

        # Calculates matrix of reaction rates
        for idx, reac in enumerate(self.model.reactions_bidirectional):
            rate_reac = reac['rate']
            for p in self.param_dict:
                rate_reac = rate_reac.subs(p, self.param_dict[p])
            variables = [atom for atom in rate_reac.atoms(sympy.Symbol) if not re.match(r'\d', str(atom))]
            func = sympy.lambdify(variables, rate_reac, modules=dict(sqrt=numpy.lib.scimath.sqrt))
            args = [y[str(l)] for l in variables]  # arguments to put in the lambdify function
            react_rate = func(*args)
            rxns_matrix[idx] = react_rate

        # maximum an minimum reaction values at each time point
        max_all_times = [max(rxns_matrix[:, col]) for col in range(numpy.shape(rxns_matrix)[1])]
        # min_all_times = [min(rxns_matrix[:, col]) for col in range(numpy.shape(rxns_matrix)[1])]

        # max_abs_flux = numpy.array([max(i, abs(j)) for i, j in zip(max_all_times, min_all_times)])

        # TODO check global_min_rate, is it the integration error? or what is the minimum rate value that makes sense
        global_min_rate = -6
        global_max_rate = magnitude(numpy.abs(max(max_all_times)))

        for i, rxn in enumerate(rxns_matrix):
            react_rate = rxn + self.mach_eps
            react_rate_magnitudes = magnitude(numpy.abs(react_rate))
            react_rate_diff_area = react_rate / numpy.power(numpy.array([10] * len(react_rate)), react_rate_magnitudes)
            rate_colors = f2hex_edges(react_rate_diff_area)
            rate_sizes = range_normalization(react_rate_magnitudes, min_x=global_min_rate, max_x=global_max_rate)
            for rctan in self.model.reactions_bidirectional[i]['reactants']:
                for pro in self.model.reactions_bidirectional[i]['products']:
                    edges_id = self.edge_name2id['s' + str(rctan) + ',s' + str(pro)]
                    all_rate_colors[edges_id] = rate_colors
                    all_rate_sizes[edges_id] = rate_sizes

        all_colors = pandas.DataFrame(all_rate_colors).transpose()
        all_sizes = pandas.DataFrame(all_rate_sizes).transpose()

        self.size_time_edges = all_sizes
        self.colors_time_edges = all_colors
        return

    # TODO CHANGE NODE SIZE INSTEAD OF NODE COLOR TO REFLECT CHANGES IN THE CONCENTRATION

    def nodes_colors(self, y):
        """

        :param y: Solution of the differential equations (from odesolve)
        :return: Returns a pandas data frame where each row is a node and each column a time point, with the color value
        """
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
        """

        :return: Creates Networkx graph from PySB model
        """

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
        """

        :param g: Networkx graph
        :return: Sets up Cytoscape graph from Networkx attributes
        """
        pos = nx.drawing.nx_agraph.pygraphviz_layout(g, prog='dot', args="-Grankdir=LR")
        g_cy = cy.network.create_from_networkx(g)
        g_cy.add_node('t')
        g_cy.add_node('Play')
        g_cy.add_node('Stop')
        g_cy.add_node('Restart')
        view_id_list = g_cy.get_views()
        self.view1 = g_cy.get_view(view_id_list[0], format='view')

        # Visual style
        minimal_style = cy.style.create('PySB')
        node_defaults = {'NODE_BORDER_WIDTH': 0}
        minimal_style.update_defaults(node_defaults)
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

        self.view1.update_network_view(visual_property='NETWORK_CENTER_X_LOCATION', value=500)
        self.view1.update_network_view(visual_property='NETWORK_CENTER_Y_LOCATION', value=0)
        self.view1.update_network_view(visual_property='NETWORK_SCALE_FACTOR', value=1)

        self.view1.update_node_views(visual_property='NODE_X_LOCATION', values=node_x_values)
        self.view1.update_node_views(visual_property='NODE_Y_LOCATION', values=node_y_values)

        # Setting up the time, play and stop node (location, fontsize, node size)
        self.view1.update_node_views(visual_property='NODE_X_LOCATION',
                                     values={self.node_name2id['t']: 50, self.node_name2id['Play']: 200,
                                             self.node_name2id['Stop']: 300, self.node_name2id['Restart']: 400})
        self.view1.update_node_views(visual_property='NODE_Y_LOCATION',
                                     values={self.node_name2id['t']: 0, self.node_name2id['Play']: 0,
                                             self.node_name2id['Stop']: 0, self.node_name2id['Restart']: 0})
        self.view1.update_node_views(visual_property='NODE_LABEL_FONT_SIZE',
                                     values={self.node_name2id['t']: 30, self.node_name2id['Play']: 30,
                                             self.node_name2id['Stop']: 30, self.node_name2id['Restart']: 30})
        self.view1.update_node_views(visual_property='NODE_TRANSPARENCY',
                                     values={self.node_name2id['t']: 0})
        self.view1.update_node_views(visual_property='NODE_BORDER_WIDTH',
                                     values={self.node_name2id['Play']: 1, self.node_name2id['Stop']: 1,
                                             self.node_name2id['Restart']: 1})
        self.view1.update_node_views(visual_property='NODE_LABEL',
                                     values={self.node_name2id['Play']: 'Play', self.node_name2id['Stop']: 'Stop',
                                             self.node_name2id['Restart']: 'Restart'})
        self.view1.update_node_views(visual_property='NODE_WIDTH',
                                     values={self.node_name2id['Play']: 70, self.node_name2id['Stop']: 70,
                                             self.node_name2id['Restart']: 70})
        self.view1.update_node_views(visual_property='NODE_HEIGHT',
                                     values={self.node_name2id['Play']: 70, self.node_name2id['Stop']: 70,
                                             self.node_name2id['Restart']: 70})

        self.view1.update_node_views(visual_property='NODE_LABEL', values=node_label_values)
        self.view1.update_node_views(visual_property='NODE_FILL_COLOR', values=node_color_values)
        self.view1.update_edge_views(visual_property='EDGE_SOURCE_ARROW_SHAPE', values=edge_source_arrow_head)
        self.view1.update_edge_views(visual_property='EDGE_TARGET_ARROW_SHAPE', values=edge_target_arrow_head)
        return

    def update_network(self, edge_color, edge_size, time_stamp=None):
        """

        :param edge_color: Dictionary where the keys are SUIDs and values are the colors assigned to edges at each time point
        :param edge_size: Dictionary where the keys are SUIDs and values are the sizes assigned to edges at each time point
        :param time_stamp: Dictionary SUID of the node that has the time label
        :return: Updates Cytoscape network with parameter values
        """
        self.view1.update_edge_views(visual_property='EDGE_WIDTH', values=edge_size)
        self.view1.update_edge_views(visual_property='EDGE_TOOLTIP', values=edge_size)
        self.view1.update_edge_views(visual_property='EDGE_STROKE_UNSELECTED_PAINT', values=edge_color)
        self.view1.update_edge_views(visual_property='EDGE_SOURCE_ARROW_UNSELECTED_PAINT', values=edge_color)
        self.view1.update_edge_views(visual_property='EDGE_TARGET_ARROW_UNSELECTED_PAINT', values=edge_color)

        if time_stamp is not None:
            self.view1.update_node_views(visual_property='NODE_LABEL', values=time_stamp)


def run_flux_visualization(model, tspan, parameters=None, verbose=False):
    """

    :param model: PySB model to visualize
    :param tspan: Time span of the simulation
    :param parameters: Model parameter values
    :param verbose: Verbose
    :return: Returns flux visualization of the PySB model
    """
    fv = FluxVisualization(model)
    fv.visualize(tspan, parameters, verbose)
