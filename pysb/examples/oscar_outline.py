# This is "stubbed" outline
# The return types in terms of arrays / dictionaries are well defined
# Don't change this file unless you've though *carefully* about the impact, and
# the information flow through the system. 
# Don't write over this file, keep it as a reference as you create the real functions.
# Using any of these functions, you can skip around in tasks orders, as they 
# return the expected results for the tyson model.
# Use the output of each to compare with the functions you create

def imposed_distance(model, t, ignore=15):
    return [2.76344e-20, 1.76761e-08, None, None,
            1.41981e-08, 0.496632, 0.00933975, 0.00525318]

# Not working, fix.....
def find_slaves(model, t, ignore=15, epsilon=1e-6):
    distance = imposed_distance(model, t, ignore)
    slaves = []
    for i,d in enumerate(distance):
        if d < epsilon:
            slaves.append('s%d'%i)
    #return slaves
    return ['s0', 's1', 's4']

# The output type may change, as needed for a graph package
# Large time (interacting with BNG)
def node_edges(model):
    # Stubbed model extraction of node edges from BNG
    # Each edge pair is directed, (0, 4) represents s0 -> s4
    return  [
                (0, 4),
                (4, 0),
                (4, 5),
                (1, 5),
                (2, 1),
                (1, 3),
                (5, 6),
                (6, 5),
                (6, 7),
                (6, 0)
            ]

def cycles(model):
    edges = node_edges(model)
    # Compute cycles using graph theory given node_edges here
    # Stubbed computation, ordering or array uncertain, symmetric under rotation
    # Numbers are equation/species number, i.e. 5 => s5
    c_list = [
                [5, 6],
                [0, 4],
                [0, 4, 5, 6]
             ]
    return c_list

def mass_conserved(model):
    edges = node_edges(model)
    # Use edges to explore equations, and build list of mass conserved equations

    # This is a stubbed computation
    mc =   [
                Symbol('s0')+Symbol('s4')+Symbol('s5')+Symbol('s6')
           ]
    return mc

# Might need a "Prune" equation function

# Large time sink, tropicalization step is needed in here, i.e. maximum
def slave_equations(model, t, ignore=15, epsilon=1e-6):
    eq = model.odes
    slaves = find_slaves(model, t, ignore, epsilon)
    conservation = mass_conserved(model)
    slave_eq = {}
    # Solve the slave equations here
    # Stubbed computation
    slave_eq = {
                 's0': (Symbol('C1')-Symbol('s5')-Symbol('s6'))*Symbol('k9')/(Symbol('k8')+Symbol('k9')),
                 's1': Symbol('k1')*(Symbol('k8')+Symbol('k9'))/(Symbol('k3')*Symbol('k8')*(Symbol('C1')-Symbol('s5')-Symbol('s6'))),
                 's4': (Symbol('C1')-Symbol('s5')-Symbol('s6'))*Symbol('k8')/(Symbol('k8')+Symbol('k9'))
               } # Stub
    return slave_eq
