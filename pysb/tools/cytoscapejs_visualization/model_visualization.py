from collections import OrderedDict
import networkx as nx
import json
from .util_networkx import from_networkx
import collections
import sympy
import numpy
from pysb.simulator.base import SimulatorException
from pysb.simulator import ScipyOdeSimulator
import pysb
import re
import matplotlib.colors as colors
import matplotlib.cm as cm

# try:
#     import matplotlib.colors as colors
# except ImportError:
#     colors = None
#
# try:
#     import matplotlib.cm as cm
# except ImportError:
#     cm = None


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


def parse_name(spec):
    """
    Function that writes short names of the species to name the nodes.
    It counts how many times a monomer_pattern is present in the complex pattern an its states
    then it takes only the monomer name and its state to write a shorter name to name the nodes.

    Parameters
    ----------
    spec : pysb.ComplexPattern
        Name of species to parse

    Returns
    Parsed name of species
    -------

    """
    m = spec.monomer_patterns
    lis_m = []
    name_counts = OrderedDict()
    parsed_name = ''
    for i in range(len(m)):
        tmp_1 = str(m[i]).partition('(')
        tmp_2 = re.findall(r"(?<=\').+(?=\')", str(m[i]))

        if not tmp_2:
            lis_m.append(tmp_1[0])
        else:
            lis_m.append(''.join([tmp_1[0], tmp_2[0]]))

    for name in lis_m:
        name_counts[name] = lis_m.count(name)

    for sp, counts in name_counts.items():
        if counts == 1:
            parsed_name += sp + '_'
        else:
            parsed_name += str(counts) + sp + '_'
    return parsed_name[:len(parsed_name) - 1]


def find_nonimportant_nodes(model):
    """
    Function that finds nodes that only have one incoming and outgoing edge

    Parameters
    ----------
    model: pysb.Model

    Returns
    a list of nodes that only have one incoming and outgoing edge
    -------

    """
    if not model.odes:
        pysb.bng.generate_equations(model)

    # gets the reactant and product species in the reactions
    rcts_sp = sum([i['reactants'] for i in model.reactions_bidirectional], ())
    pdts_sp = sum([i['products'] for i in model.reactions_bidirectional], ())
    # find the reactants and products that are only used once
    non_imp_rcts = set([x for x in range(len(model.species)) if rcts_sp.count(x) < 2])
    non_imp_pdts = set([x for x in range(len(model.species)) if pdts_sp.count(x) < 2])
    non_imp_nodes = set.intersection(non_imp_pdts, non_imp_rcts)
    passengers = non_imp_nodes
    return passengers


class ModelVisualization(object):
    """
    Class to visualize models and their dynamics

    Parameters
    ----------
    model : pysb.Model
        Model to visualize.
    """
    mach_eps = numpy.finfo(float).eps

    def __init__(self, model):

        self.model = model
        self.tspan = None
        self.y_df = None
        self.param_dict = None
        self.sp_graph = None
        self.passengers = []
        self.is_setup = False

    def static_view(self, get_passengers=True):
        """
        Generates a dictionary with the model graph data that can be converted in the Cytoscape.js JSON format

        Parameters
        ----------
        get_passengers : bool
            if True, nodes that only have one incoming or outgoing edge are painted with a different color

        Examples
        --------
        >>> from pysb.examples.earm_1_0 import model
        >>> import numpy as np
        >>> viz = ModelVisualization(model)
        >>> data = viz.static_view()

        Returns
        -------
        A Dictionary Object that can be converted into Cytoscape.js JSON
        """
        if get_passengers:
            self.passengers = find_nonimportant_nodes(self.model)
        self.species_graph(view='static')
        g_layout = self.dot_layout(self.sp_graph)
        data = self.graph_to_json(sp_graph=self.sp_graph, layout=g_layout)
        return data

    def dynamic_view(self, tspan=None, param_values=None, get_passengers=True, verbose=False):
        """
        Generates a dictionary with the model dynamics data that can be converted in the Cytoscape.js JSON format

        Parameters
        ----------
        tspan : vector-like, optional
            Time values over which to simulate and visualize
        param_values : vector-like or dict, optional
            Values to use for every parameter in the model. Ordering is
            determined by the order of model.parameters.
            If passed as a dictionary, keys must be parameter names.
            If not specified, parameter values will be taken directly from
            model.parameters.
        get_passengers : bool
            if True, nodes that only have one incoming or outgoing edge are painted with a different color
        verbose : bool
            if True, prints state of code execution

        Examples
        --------
        >>> from pysb.examples.earm_1_0 import model
        >>> import numpy as np
        >>> viz = ModelVisualization(model)
        >>> data = viz.dynamic_view(tspan=np.linspace(0, 20000, 100))

        Returns
        -------
        A Dictionary Object that can be converted into Cytoscape.js JSON
        """
        # if (colors is None) or (cm is None):
        #     raise Exception('Please "pip install matplotlib" for this feature')

        self._setup_dynamics(tspan=tspan, param_values=param_values, get_passengers=get_passengers, verbose=verbose)
        self.species_graph(view='dynamic')
        g_layout = self.dot_layout(self.sp_graph)
        self._add_edge_node_dynamics()
        data = self.graph_to_json(sp_graph=self.sp_graph, layout=g_layout)
        return data

    def _setup_dynamics(self, tspan=None, param_values=None, get_passengers=True, verbose=False):
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
            self.passengers = find_nonimportant_nodes(self.model)

        # Solution of the differential equations
        sim_result = ScipyOdeSimulator(self.model, tspan=self.tspan, param_values=self.param_dict).run()
        self.y_df = sim_result.all

        self.is_setup = True
        return

    def species_graph(self, view):
        """
        Creates a networkx.digraph graph

        Parameters
        ----------
        view : str
            Name of the type of visualization, it can either 'static' or 'dynamic'

        Returns
        -------
        A :class: nx.Digraph graph that has the information for the visualization of the model
        """
        if view == 'static':
            self.sp_graph = OrderedGraph(name=self.model.name, view=view)
        elif view == 'dynamic':
            self.sp_graph = OrderedGraph(name=self.model.name, tspan=self.tspan.tolist(), view=view)
        else:
            raise ValueError('View is not valid')
        # TODO: there are reactions that generate parallel edges that are not taken into account because netowrkx
        # digraph only allows one edge between two nodes

        for idx, cp in enumerate(self.model.species):
            species_node = 's%d' % idx
            # Setting the information about the node
            node_data = {}
            node_data['label'] = parse_name(self.model.species[idx])
            if idx in self.passengers:
                node_data['background_color'] = "#162899"
            else:
                node_data['background_color'] = "#2b913a"

            self.sp_graph.add_node(species_node, **node_data)

        for reaction in self.model.reactions_bidirectional:
            reactants = set(reaction['reactants'])
            products = set(reaction['products'])
            attr_reversible = {'source_arrow_shape': 'diamond', 'target_arrow_shape': 'triangle'} \
                                if reaction['reversible'] \
                                else {'source_arrow_shape': 'none', 'target_arrow_shape': 'triangle'}
            for s in reactants:
                for p in products:
                    self._r_link(s, p, **attr_reversible)
        return self.sp_graph

    def _r_link(self, s, r, **attrs):
        """
        Links two nodes in a species graph
        Parameters
        ----------
        s : str
            Source node
        r: str
            Target node
        attrs: dict
            Other attributes for edges

        Returns
        -------

        """

        nodes = ('s%d' % s, 's%d' % r)
        if attrs.get('_flip'):
            del attrs['_flip']
            nodes = nodes[::-1]
        # attrs.setdefault('arrowhead', 'normal')
        self.sp_graph.add_edge(*nodes, **attrs)

    def _add_edge_node_dynamics(self):
        """
        Add the edge size and color data as well as node color and values data
        Returns
        -------

        """
        # in networkx 2.0 the order is G, values, names
        # in networkx 1.11 the order is G, names, values
        if float(nx.__version__) >= 2:
            edge_sizes, edge_colors, edge_qtips = self.edges_colors_sizes()
            nx.set_edge_attributes(self.sp_graph, edge_colors, 'edge_color')
            nx.set_edge_attributes(self.sp_graph, edge_sizes, 'edge_size')
            nx.set_edge_attributes(self.sp_graph, edge_qtips, 'edge_qtip')

            node_abs, node_rel = self.node_data()
            nx.set_node_attributes(self.sp_graph, node_abs, 'abs_value')
            nx.set_node_attributes(self.sp_graph, node_rel, 'rel_value')
        else:
            edge_sizes, edge_colors, edge_qtips = self.edges_colors_sizes()
            nx.set_edge_attributes(self.sp_graph, 'edge_color', edge_colors)
            nx.set_edge_attributes(self.sp_graph, 'edge_size', edge_sizes)
            nx.set_edge_attributes(self.sp_graph, 'edge_qtip', edge_qtips)

            node_abs, node_rel = self.node_data()
            nx.set_node_attributes(self.sp_graph, 'abs_value', node_abs)
            nx.set_node_attributes(self.sp_graph, 'rel_value', node_rel)

    @staticmethod
    def graph_to_json(sp_graph, layout=None, path=''):
        """

        Parameters
        ----------
        sp_graph : nx.Digraph graph
            A graph to be converted into cytoscapejs json format
        layout: str
            Name of the layout algorithm to use for the visualization
        path: str
            Path to save the file

        Returns
        -------
        A Dictionary Object that can be converted into Cytoscape.js JSON
        """
        data = from_networkx(sp_graph, layout=layout, scale=1)
        if path:
            with open(path + 'data.json', 'w') as outfile:
                json.dump(data, outfile)
        return data

    @staticmethod
    def dot_layout(sp_graph):
        """

        Parameters
        ----------
        sp_graph : nx.Digraph graph
            Graph to layout

        Returns
        -------
        An OrderedDict containing the node position according to the dot layout
        """

        pos = nx.drawing.nx_agraph.pygraphviz_layout(sp_graph, prog='dot', args="-Grankdir=LR")
        ordered_pos = collections.OrderedDict((node, pos[node]) for node in sp_graph.nodes())
        return ordered_pos

    def edges_colors_sizes(self):
        """

        Returns
        -------
        This function obtains values for the size and color of the edges in the network.
        The color is a representation of the percentage of flux going through an edge.
        The edge size is a representation of the relative value of the reaction normalized to the maximum value that
        the edge can attain during the whole simulation.
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

        vals_norm = numpy.vectorize(self.mon_normalized)
        all_products = [rx['products'] for rx in self.model.reactions_bidirectional]
        all_reactants = [rx['reactants'] for rx in self.model.reactions_bidirectional]
        sp_imp = set(range(len(self.model.species))) - self.passengers  # species with more than one edge

        for sp in sp_imp:
            # Getting the indices of the reactions_bidirectional in which sp is a reactant
            rxns_idx_c = [all_reactants.index(rx) for rx in all_reactants if sp in rx]

            rxn_val_pos = numpy.where(rxns_matrix[rxns_idx_c] > 0, rxns_matrix[rxns_idx_c], 0).sum(axis=0)
            rxn_val_neg = numpy.where(rxns_matrix[rxns_idx_c] < 0, rxns_matrix[rxns_idx_c], 0).sum(axis=0)
            rxn_val_total = rxns_matrix[rxns_idx_c].sum(axis=0)
            for rx in rxns_idx_c:
                products = all_products[rx]
                for p in products:
                    react_rate_color = rxns_matrix[rx] / rxn_val_pos
                    rxn_neg_idx = numpy.where(react_rate_color < 0)
                    react_rate_color[rxn_neg_idx] = -1 * (react_rate_color[rxn_neg_idx] / rxn_val_neg[rxn_neg_idx])
                    numpy.nan_to_num(react_rate_color, copy=False)
                    rate_colors = self.f2hex_edges(react_rate_color)

                    rxn_eps = rxns_matrix[rx] + self.mach_eps
                    rxn_max = rxn_eps.max()
                    rxn_min = abs(rxn_eps.min())
                    react_rate_size = vals_norm(rxn_eps, rxn_max, rxn_min)
                    rate_sizes = self.range_normalization(numpy.abs(react_rate_size), min_x=0, max_x=1)
                    edges_id = ('s' + str(sp), 's' + str(p))
                    all_rate_colors[edges_id] = rate_colors
                    all_rate_sizes[edges_id] = rate_sizes.tolist()
                    all_rate_abs_val[edges_id] = rxns_matrix[rx].tolist()

        for sp in self.passengers:
            rxns_idx_c = [all_reactants.index(rx) for rx in all_reactants if sp in rx]

            rxn_val_total = rxns_matrix[rxns_idx_c].sum(axis=0)
            for rx in rxns_idx_c:
                products = all_products[rx]
                for p in products:
                    edges_id = ('s' + str(sp), 's' + str(p))
                    all_rate_colors[edges_id] = ['#A4A09F'] * len(self.tspan)
                    all_rate_sizes[edges_id] = [0.5] * len(self.tspan)
                    all_rate_abs_val[edges_id] = rxns_matrix[rx].tolist()

        # self.size_time_edges = all_rate_sizes
        # self.colors_time_edges = all_rate_colors
        # self.rxn_abs_vals = all_rate_abs_val

        return all_rate_sizes, all_rate_colors, all_rate_abs_val

    def node_data(self):
        """
        Converts concentration values to colors to be used in the nodes
        :return: Returns a pandas data frame where each row is a node and each column a time point, with the color value
        """
        node_absolute = {}
        node_relative = {}
        for sp in range(len(self.model.species)):
            sp_absolute = self.y_df['__s%d' % sp]
            sp_relative = (sp_absolute / sp_absolute.max()) * 100
            node_absolute['s%d' % sp] = sp_absolute.tolist()
            node_relative['s%d' % sp] = sp_relative.tolist()

        # all_nodes_values = pandas.DataFrame(all_rate_colors)
        return node_absolute, node_relative

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

            colors_hex[i] = '#{0:02x}{1:02x}{2:02x}'.format(*tuple([int(255 * fc) for fc in color]))
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
