from collections import OrderedDict
import networkx as nx
from pysb.tools.helper_functions import parse_name
import json
from py2cytoscape.util import from_networkx
import collections
import sympy
import numpy
import matplotlib.colors as colors
import matplotlib.cm as cm
from pysb.simulator.base import SimulatorException
from pysb.simulator import ScipyOdeSimulator
import tropical.helper_functions as hf


class OrderedGraph(nx.DiGraph):
    """
    Networkx Digraph that stores the nodes in the order they are input
    """
    node_dict_factory = OrderedDict
    adjlist_outer_dict_factory = OrderedDict
    adjlist_inner_dict_factory = OrderedDict
    edge_attr_dict_factory = OrderedDict


class MidpointNormalize(colors.Normalize):
    """
    A class which, when called, can normalize data into the ``[vmin,midpoint,vmax] interval
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return numpy.ma.masked_array(numpy.interp(value, x, y))


class FluxVisualization:
    mach_eps = numpy.finfo(float).eps

    def __init__(self, model):
        self.model = model
        self.tspan = None
        self.y_df = None
        self.param_dict = None
        self.sp_graph = None
        self.size_time_edges = None
        self.colors_time_edges = None
        self.rxn_abs_vals = None
        self.passengers = []
        self.is_setup = False

    def setup_info(self, tspan=None, param_values=None, get_passengers=False, verbose=False):
        if verbose:
            print("Solving Simulation")

        if tspan is not None:
            self.tspan = tspan
        elif self.tspan is None:
            raise SimulatorException("'tspan' must be defined.")

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

        # Gets passenger species to visualize in the network
        if get_passengers:
            self.passengers = hf.find_nonimportant_nodes(self.model)

        # Solution of the differential equations
        sim_result = ScipyOdeSimulator(self.model, tspan=self.tspan, param_values=self.param_dict).run()
        self.y_df = sim_result.all

        self.edges_colors_sizes()
        self.is_setup = True
        return

    def species_graph(self):
        """
        Creates a species graph
        :return: Creates Networkx graph from PySB model, it includes the results of simulation, where edges have
        different sizes for different reactions rates and nodes have different values
        """
        # TODO: there are reactions that generate parallel edges that are not taken into account because netowrkx
        # digraph only allows one edge between two nodes
        self.sp_graph = OrderedGraph(name=self.model.name, tspan=self.tspan.tolist())
        for idx, cp in enumerate(self.model.species):
            species_node = 's%d' % idx
            node_data = self.node_relative_values(idx)
            node_data['label'] = parse_name(self.model.species[idx])
            node_data['shape_cy'] = "roundrectangle"
            node_data['font_size'] = 18
            if idx in self.passengers:
                node_data['background_color'] = "#162899"
            else:
                node_data['background_color'] = "#2b913a"

            self.sp_graph.add_node(species_node,
                                   node_data)

        for reaction in self.model.reactions_bidirectional:
            reactants = set(reaction['reactants'])
            products = set(reaction['products'])
            attr_reversible = {'source_arrow_shape': 'diamond', 'target_arrow_shape': 'triangle',
                               'source_arrow_fill': 'filled', 'width': 3, 'curve_style': 'bezier'} if reaction[
                'reversible'] else {
                'source_arrow_shape': 'none', 'target_arrow_shape': 'triangle', 'width': 6}
            for s in reactants:
                for p in products:
                    self._r_link(s, p, **attr_reversible)
        return self.sp_graph

    def _r_link(self, s, r, **attrs):
        """
        Links two nodes in a species graph
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
        link_name = ','.join(i for i in nodes)
        attrs['name'] = link_name
        for idx in range(len(self.tspan)):
            attrs['edge_color_t{0}'.format(idx)] = self.colors_time_edges[link_name][idx]
            attrs['edge_size_t{0}'.format(idx)] = self.size_time_edges[link_name][idx]
            attrs['edge_qtip_t{0}'.format(idx)] = self.rxn_abs_vals[link_name][idx]
        # attrs['edges_colors'] = self.colors_time_edges[link_name].values.tolist()
        # attrs['edges_sizes'] = self.size_time_edges[link_name].values.tolist()
        self.sp_graph.add_edge(*nodes, **attrs)

    def graph_to_json(self, path='', layout=None):
        if not self.sp_graph:
            self.species_graph()
        data = from_networkx(self.sp_graph, layout=layout, scale=1)
        with open(path + 'data.json', 'w') as outfile:
            json.dump(data, outfile)

    def dot_layout(self):
        if not self.sp_graph:
            self.species_graph()
        pos = nx.drawing.nx_agraph.pygraphviz_layout(self.sp_graph, prog='dot', args="-Grankdir=LR")
        # TODO: may be better to change the way the py2cytoscape function reads the layout
        ordered_pos = collections.OrderedDict((node, pos[node]) for node in self.sp_graph.nodes())
        return ordered_pos

    def edges_colors_sizes(self):
        """
        Converts reaction rates values to colors and sizes for the edges
        :return: Returns a pandas data frame where each row is an edge and each column a time point, with the color value
        """
        all_rate_colors = {}
        all_rate_sizes = {}
        all_rate_abs_val = {}
        rxns_matrix = numpy.zeros((len(self.model.reactions_bidirectional), len(self.tspan)))

        # Calculates matrix of reaction rates
        for idx, reac in enumerate(self.model.reactions_bidirectional):
            rate_reac = reac['rate']
            for p in self.param_dict:
                rate_reac = rate_reac.subs(p, self.param_dict[p])
            variables = [atom for atom in rate_reac.atoms(sympy.Symbol)]
            args = [0] * len(variables)  # arguments to put in the lambdify function
            for idx2, va in enumerate(variables):
                if str(va).startswith('__'):
                    args[idx2] = self.y_df[str(va)]
                else:
                    args[idx2] = self.param_dict[va.name]
            func = sympy.lambdify(variables, rate_reac, modules=dict(sqrt=numpy.lib.scimath.sqrt))
            react_rate = func(*args)
            rxns_matrix[idx] = react_rate

        # maximum an minimum reaction values at each time point
        # max_all_times = [max(rxns_matrix[:, col]) for col in range(numpy.shape(rxns_matrix)[1])]
        # min_all_times = [min(rxns_matrix[:, col]) for col in range(numpy.shape(rxns_matrix)[1])]

        # max_abs_flux = numpy.array([max(i, abs(j)) for i, j in zip(max_all_times, min_all_times)])
        vals_norm = numpy.vectorize(self.mon_normalized)
        all_products = [rx['products'] for rx in self.model.reactions_bidirectional]
        all_reactants = [rx['reactants'] for rx in self.model.reactions_bidirectional]

        for sp in range(len(self.model.species)):
            rxns_idx_p = [all_products.index(rx) for rx in all_products if sp in rx]

            rxn_val_total = rxns_matrix[rxns_idx_p].sum(axis=0)
            for rx in rxns_idx_p:
                reactants = self.model.reactions_bidirectional[rx]['reactants']
                for r in reactants:
                    react_rate_color = rxns_matrix[rx] / rxn_val_total
                    numpy.nan_to_num(react_rate_color, copy=False)
                    rate_colors = self.f2hex_edges(react_rate_color)

                    rxn_eps = rxns_matrix[rx] + self.mach_eps
                    rxn_max = rxn_eps.max()
                    rxn_min = abs(rxn_eps.min())
                    react_rate_size = vals_norm(rxn_eps, rxn_max, rxn_min)
                    rate_sizes = self.range_normalization(numpy.abs(react_rate_size), min_x=0, max_x=1)
                    edges_id = 's' + str(r) + ',s' + str(sp)
                    all_rate_colors[edges_id] = rate_colors
                    all_rate_sizes[edges_id] = rate_sizes
                    all_rate_abs_val[edges_id] = rxns_matrix[rx]

        # for i, rxn in enumerate(rxns_matrix):
        #     rxn_eps = rxn + self.mach_eps
        #     rxn_max = rxn_eps.max()
        #     rxn_min = abs(rxn_eps.min())
        #     react_rate_norm = vals_norm(rxn_eps, rxn_max, rxn_min)
        #     rate_colors = self.f2hex_edges(react_rate_norm)
        #     rate_sizes = self.range_normalization(numpy.abs(react_rate_norm), min_x=0, max_x=1)
        #     for rctan in self.model.reactions_bidirectional[i]['reactants']:
        #         for pro in self.model.reactions_bidirectional[i]['products']:
        #             edges_id = 's' + str(rctan) + ',s' + str(pro)
        #             all_rate_colors[edges_id] = rate_colors
        #             all_rate_sizes[edges_id] = rate_sizes
        #             all_rate_abs_val[edges_id] = rxn

        # all_colors = pandas.DataFrame(all_rate_colors)
        # all_sizes = pandas.DataFrame(all_rate_sizes)
        self.size_time_edges = all_rate_sizes
        self.colors_time_edges = all_rate_colors
        self.rxn_abs_vals = all_rate_abs_val

        return

    def node_relative_values(self, sp):
        """
        Converts concentration values to colors to be used in the nodes
        :return: Returns a pandas data frame where each row is a node and each column a time point, with the color value
        """

        node_data = {}
        sp_values = self.y_df['__s%d' % sp]
        node_values = (sp_values / sp_values.max()) * 100
        for abs_val, rel_val, i in zip(sp_values, node_values, range(len(self.tspan))):
            node_data['abs_value_t{0}'.format(i)] = abs_val
            node_data['rel_value_t{0}'.format(i)] = rel_val
        # all_nodes_values = pandas.DataFrame(all_rate_colors)
        return node_data

    @staticmethod
    def f2hex_edges(fx, vmin=-0.99, vmax=0.99):
        """
        Converts reaction rates values to f2hex colors
        :param fx: Vector of reaction rates (flux)
        :param vmin: Value of minimum for normalization
        :param vmax: Value of maximum for normalization
        :return: This function returns a vector of colors in hex format that represents flux
        """
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
        f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('RdBu_r'))
        rgb = [f2rgb.to_rgba(rate)[:3] for rate in fx]
        colors_hex = [0] * (len(rgb))
        for i, color in enumerate(rgb):
            colors_hex[i] = '#%02x%02x%02x' % tuple([255 * fc for fc in color])
        return colors_hex

    @staticmethod
    def range_normalization(x, min_x, max_x, a=0.5, b=10):
        """
        Normalized vector to the [0.1,15] range
        :param x: Vector of numbers to be normalized
        :param min_x: Minimum value in vector x
        :param max_x: Maximum value in vector x
        :param a: Value of minimum used for the normalization
        :param b: Value of maximum used for the normalization
        :return: Normalized vector
        """
        return a + (x - min_x) * (b - a) / (max_x - min_x)

    @staticmethod
    def mon_normalized(x, max_value, min_value):
        if x > 0:
            return x / max_value
        else:
            return x / min_value
