from pysb import ComponentSet
import pysb.core
import inspect
import numpy
import io
import networkx as nx
__all__ = ['alias_model_components', 'rules_using_parameter', 'read_dot']


def alias_model_components(model=None):
    """Make all model components visible as symbols in the caller's global namespace"""
    if model is None:
        model = pysb.core.SelfExporter.default_model
    caller_globals = inspect.currentframe().f_back.f_globals
    components = dict((c.name, c) for c in model.all_components())
    caller_globals.update(components)


def rules_using_parameter(model, parameter):
    """Return a ComponentSet of rules in the model which make use of the given parameter"""
    if not isinstance(parameter, pysb.core.Parameter):
        # Try getting the parameter by name
        parameter = model.parameters.get(parameter)
    cset = ComponentSet()
    for rule in model.rules:
        if rule.rate_forward is parameter or rule.rate_reverse is parameter:
            cset.add(rule)
    return cset


def synthetic_data(model, tspan, obs_list=None, sigma=0.1):
    #from pysb.integrate import odesolve
    from pysb.integrate import Solver
    solver = Solver(model, tspan)
    solver.run()

    # Sample from a normal distribution with variance sigma and mean 1
    # (randn generates a matrix of random numbers sampled from a normal
    # distribution with mean 0 and variance 1)
    # 
    # Note: This modifies yobs_view (the view on yobs) so that the changes 
    # are reflected in yobs (which is returned by the function). Since a new
    # Solver object is constructed for each function invocation this does not
    # cause problems in this case.
    solver.yobs_view *= ((numpy.random.randn(*solver.yobs_view.shape) * sigma) + 1)
    return solver.yobs


def get_param_num(model, name):
    for i in range(len(model.parameters)):
        if model.parameters[i].name == name:
            print(i, model.parameters[i])
            break
    return i


def write_params(model,paramarr, name=None):
    """ write the parameters and values to a csv file
    model: a model object
    name: a string with the name for the file, or None to return the content
    """
    if name is not None:
        fobj = open(name, 'w')
    else:
        fobj = io.StringIO()
    for i in range(len(model.parameters)):
        fobj.write("%s, %.17g\n"%(model.parameters[i].name, paramarr[i]))
    if name is None:
        return fobj.getvalue()


def update_param_vals(model, newvals):
    """update the values of model parameters with the values from a dict. 
    the keys in the dict must match the parameter names
    """
    update = []
    noupdate = []
    for i in model.parameters:
        if i.name in newvals:
            i.value = newvals[i.name]
            update.append(i.name)
        else:
            noupdate.append(i.name)
    return update, noupdate


def load_params(fname):
    """load the parameter values from a csv file, return them as dict.
    """
    parmsff = {}
    # FIXME: This might fail if a parameter name is larger than 50 characters.
    # FIXME: Maybe do this with the csv module instead?
    temparr = numpy.loadtxt(fname, dtype=([('a','S50'),('b','f8')]), delimiter=',') 
    for i in temparr:
        parmsff[i[0]] = i[1]
    return parmsff


def read_dot(filename):
    """ Read a graphviz dot file using pydot

    Parameters
    ----------
    filename: str
        A DOT (graphviz) filename

    Returns
    -------
    MultiGraph
        A networkx MultiGraph file
    """
    try:
        import pydot
        pydot_graph = pydot.graph_from_dot_file(filename)[0]
        return _from_pydot(pydot_graph)
    except ImportError:
        raise ImportError('Importing graphviz files requires the pydot '
                          'library')


def _from_pydot(P):
    """Return a NetworkX graph from a Pydot graph.

    Using this patched version until networkx issue is resolved, which fixes
    .dot file support using PyDot:
    https://github.com/networkx/networkx/issues/2832

    Parameters
    ----------

    P : Pydot graph
      A graph created with Pydot

    Returns
    -------
    G : NetworkX multigraph
        A MultiGraph or MultiDiGraph.

    Examples
    --------
    >>> K5 = nx.complete_graph(5)
    >>> A = nx.nx_pydot.to_pydot(K5)
    >>> G = nx.nx_pydot.from_pydot(A) # return MultiGraph

    # make a Graph instead of MultiGraph
    >>> G = nx.Graph(nx.nx_pydot.from_pydot(A))

    """
    if P.get_strict(None):  # pydot bug: get_strict() shouldn't take argument
        multiedges = False
    else:
        multiedges = True

    if P.get_type() == 'graph':  # undirected
        if multiedges:
            N = nx.MultiGraph()
        else:
            N = nx.Graph()
    else:
        if multiedges:
            N = nx.MultiDiGraph()
        else:
            N = nx.DiGraph()

    # assign defaults
    name = P.get_name().strip('"')
    if name != '':
        N.name = name

    # add nodes, attributes to N.node_attr
    for p in P.get_node_list():
        n = p.get_name().strip('"')
        if n in ('node', 'graph', 'edge'):
            continue
        N.add_node(n, **{k: v.strip('"')
                         for k, v in p.get_attributes().items()})

    # add edges
    for e in P.get_edge_list():
        u = e.get_source()
        v = e.get_destination()
        attr = {k: v.strip('"') for k, v in e.get_attributes().items()}
        s = []
        d = []

        if isinstance(u, str):
            s.append(u.strip('"'))
        else:
            for unodes in u['nodes']:
                s.append(unodes.strip('"'))

        if isinstance(v, str):
            d.append(v.strip('"'))
        else:
            for vnodes in v['nodes']:
                d.append(vnodes.strip('"'))

        for source_node in s:
            for destination_node in d:
                N.add_edge(source_node, destination_node, **attr)

    # add default attributes for graph, nodes, edges
    pattr = {k: v.strip('"') for k, v in P.get_attributes().items()}
    if pattr:
        N.graph['graph'] = pattr
    try:
        N.graph['node'] = P.get_node_defaults()[0]
    except:  # IndexError,TypeError:
        pass  # N.graph['node']={}
    try:
        N.graph['edge'] = P.get_edge_defaults()[0]
    except:  # IndexError,TypeError:
        pass  # N.graph['edge']={}
    return N
