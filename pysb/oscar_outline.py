# This is "stubbed" outline
# The return types in terms of arrays / dictionaries are well defined
# Don't change this file unless you've though *carefully* about the impact, and
# the information flow through the system. 
# Don't write over this file, keep it as a reference as you create the real functions.
# Using any of these functions, you can skip around in tasks orders, as they 
# return the expected results for the tyson model.
# Use the output of each to compare with the functions you create

from sympy.solvers import solve
from sympy import Symbol
from sympy import symbols
from sympy import symarray
from sympy.functions.elementary.complexes import Abs
from sympy import solve_poly_system
from sympy import log
from sympy.functions.special.delta_functions import Heaviside
from sympy import simplify 
from sympy import Mul
from sympy import log
from collections import defaultdict

import pysb
import pysb.bng
import sympy
import re
import sys
import os
import pygraphviz
import networkx
import copy
from sympy.parsing.sympy_parser import parse_expr
from collections import Mapping
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sympy import lambdify


from pysb.bng import generate_equations
from pysb.integrate import odesolve
from pysb.examples.tyson_oscillator_harris import model
import numpy
from sympy import sqrt
import matplotlib.pyplot as plt


generate_equations(model)
t = numpy.linspace(0, 100, 10001)
x = odesolve(model, t)

def find_slaves(model, t, ignore=20, epsilon=1e-2):   
    slaves = []
    generate_equations(model)
    x = odesolve(model, t)
    x = x[ignore:] # Ignore first couple points
    t = t[ignore:]
    names = [n for n in filter(lambda n: n.startswith('__'), x.dtype.names)]
    x = x[names] # Only concrete species are considered This creates model.odes which contains the math
    names = [n.replace('__','') for n in names]
    x.dtype = [(n,'<f8') for n in names]
    a = [] #list of solved polynomial equations
    c = defaultdict(list)
    b = []
    for i, eq in enumerate(model.odes): # i is equation number
        eq   = eq.subs('s%d' % i, 's%dstar' % i)
        sol  = solve(eq, Symbol('s%dstar' % i)) # Find equation of imposed trace
        for j in range(len(sol)):  # j is solution j for equation i
            for p in model.parameters: sol[j] = sol[j].subs(p.name, p.value) # Substitute parameters
            a.append(sol[j]) 
            b.append(i)    
#         print i,j
    for k,e in enumerate(a):
        args = [] #arguments to put in lambdify function
        variables = [atom for atom in a[k].atoms(Symbol) if not re.match(r'\d',str(atom))]
        f = lambdify(variables, a[k], modules = dict(sqrt=numpy.lib.scimath.sqrt) )
        variables_l = list(variables)
       # print variables
        for u,l in enumerate(variables_l):
            args.append(x[:][str(l)])
        hey = abs(numpy.log(f(*args)) - numpy.log(x[:]['s%d'%b[k]]))
        if hey.max() <= epsilon : slaves.append('s%d'%b[k])
        #print hey.max()
#        c['s%d'%b[k]].append(f(*args))
    #print slaves
    
             
        
    
        
    return slaves

# The output type may change, as needed for a graph package
# Large time (interacting with BNG)

# This is a function which builds the edges according to the nodes
def r_link(graph, s, r, **attrs):
    nodes = ('s%d' % s, 's%d' % r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = reversed(nodes)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)

def find_cycles(model):
    """
    Render the reactions produced by a model into the "dot" graph format.

    Parameters
    ----------
    model : pysb.core.Model
        The model to render.

    Returns
    -------
    sorted graph edges
    """

    pysb.bng.generate_equations(model)

    graph = networkx.DiGraph(rankdir="LR")
    ic_species = [cp for cp, parameter in model.initial_conditions]
    for i, cp in enumerate(model.species):
        species_node = 's%d' % i
        slabel = re.sub(r'% ', r'%\\l', str(cp))
        slabel += '\\l'
        color = "#ccffcc"
        # color species with an initial condition differently
        if len([s for s in ic_species if s.is_equivalent_to(cp)]):
            color = "#aaffff"
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
 #   networkx.draw(graph) 
 #   plt.show() 
    return list(networkx.simple_cycles(graph)) #graph.edges() returns the edges

#This function finds conservation laws from the conserved cycles
def mass_conserved(model):
    c = find_cycles(model)
    h = []
    g = []
    for i, item in enumerate(c):
        b = 0
        u = 0
        for j, specie in enumerate(item):
            b += model.odes[int(re.findall(r'\d+', c[i][j])[0])]
        if b == 0:
            g.append(item)
            for l,k in enumerate(item):
                u += sympy.Symbol(c[i][l])    
            h.append(u-sympy.Symbol('C%d'%i))
            print 'cycle%d'%i, 'is conserved'
    #print h,g
            
    return h, g

# Might need a "Prune" equation function

# Large time sink, tropicalization step is needed in here, i.e. maximum
def slave_equations(model, t, ignore=20, epsilon=1e-2):
    slaves = find_slaves(model, t, ignore=20, epsilon=1e-2)
    slave_conserved_eqs = {}
    for i, j in enumerate(slaves):
        slave_conserved_eqs[j] = model.odes[int(re.findall(r'\d+', slaves[i])[0])] 

#        slave_conserved_eqs.setdefault(j,[]).append(model.odes[int(re.findall(r'\d+', slaves[i])[0])])
        
    # Solve the slave equations here
    # Stubbed computation
#    slave_eq = {
#                 's0': (Symbol('C1')-Symbol('s5')-Symbol('s6'))*Symbol('k9')/(Symbol('k8')+Symbol('k9')),
#                 's1': Symbol('k1')*(Symbol('k8')+Symbol('k9'))/(Symbol('k3')*Symbol('k8')*(Symbol('C1')-Symbol('s5')-Symbol('s6'))),
#                 's4': (Symbol('C1')-Symbol('s5')-Symbol('s6'))*Symbol('k8')/(Symbol('k8')+Symbol('k9'))
#               } # Stub
    return slave_conserved_eqs

def find_nearest(array,value):
    idx = (numpy.abs(array-value)).argmin()
    return array[idx]

def pruned_equations(model, t, ignore=20, epsilon=1e-2, rho=5):

    #k8, s5, k9, s0, k3, s1, k1, s2, k2, k3, k6, s6 = symbols('k8 s5 k9 s0 k3 s1 k1 s2 k2 k3 k6 s6')
    #a = [k8*s5-k9*s0, -k8*s5+k9*s0, k1*s2-k2*s1-k3*s0*s1]
    #return a

    generate_equations(model)
    x = odesolve(model, t)
    x = x[ignore:] # Ignore first couple points
    t = t[ignore:]
    names = [n for n in filter(lambda n: n.startswith('__'), x.dtype.names)]
    x = x[names] # Only concrete species are considered
    names = [n.replace('__','') for n in names]
    x.dtype = [(n,'<f8') for n in names]
    conservation = mass_conserved(model)[0]
    pruned_eqs = slave_equations(model, t, ignore=15, epsilon=1e-2)
    eq = copy.deepcopy(pruned_eqs)
    
    for i, j in enumerate(eq):
        ble = eq[j].as_coefficients_dict().keys()#Creates a list of the monomials of each slave equation
        for l, m in enumerate(ble): #Compares the monomials to find the pruned system
            m_ready = m
            m_elim = m
            for p in model.parameters: m_ready = m_ready.subs(p.name, p.value) # Substitute parameters
            for k in range(len(ble)):
                if (k+l+1) <= (len(ble)-1):
                    ble_ready = ble[k+l+1]
                    ble_elim = ble[k+l+1]
                    for p in model.parameters: ble_ready = ble_ready.subs(p.name, p.value) # Substitute parameters
                    args2 = []
                    args1 = []
                    variables_ble_ready = [atom for atom in ble_ready.atoms(Symbol) if not re.match(r'\d',str(atom))]
                    variables_m_ready = [atom for atom in m_ready.atoms(Symbol) if not re.match(r'\d',str(atom))]
                    f_ble = lambdify(variables_ble_ready, ble_ready, 'numpy' )
                    f_m = lambdify(variables_m_ready, m_ready, 'numpy' )
                    for uu,ll in enumerate(variables_ble_ready):
                        args2.append(x[:][str(ll)])
                    for w,s in enumerate(variables_m_ready):
                        args1.append(x[:][str(s)])
                    hey_pruned = f_m(*args1) - f_ble(*args2)
                    diff = find_nearest(hey_pruned, 0)
                    diff_pru = numpy.abs(diff)
                    if diff > 0 and diff_pru > rho:
                       pruned_eqs[j] = pruned_eqs[j].subs(ble_elim, 0)
                    if diff < 0 and diff_pru > rho:\
                       pruned_eqs[j] = pruned_eqs[j].subs(m_elim, 0)
    for i, l in enumerate(conservation): #Add the conservation laws to the pruned system
        pruned_eqs['cons%d'%i]=l
    return pruned_eqs


def diff_alg_system(model):
    sol_dict = {}
    index_slaves = []
    slaves = find_slaves(model, t, ignore=20, epsilon=1e-2)
    var_ready = []
    eqs_to_add = copy.deepcopy(model.odes)
    eqs_to_add_dict = {}


 
    var = find_slaves(model, t, ignore=20, epsilon=1e-2)
    eqs = pruned_equations(model, t, ignore=20, epsilon=1e-2, rho=5)
    eqs_l = []
    w = mass_conserved(model)[1]
    cycle_eqs = mass_conserved(model)[0]
    
    for i,j in enumerate(eqs):
        eqs_l.append(eqs[j])    
 
    for i in w: #Adds the variable of s2 cycle, it is required because the solver doesn't know if s2 or C2 is the constant 
        if len(i) == 1:
            var.append(i[0])
    for j in var:
        var_ready.append(Symbol(j))
    sol = solve_poly_system(eqs_l, var_ready)
    for i, j in enumerate(var_ready):
        sol_dict[j] = sol[0][i]
    print sol_dict
    for i, j in enumerate(eqs_to_add):
        eqs_to_add_dict[Symbol('s%d'%i)] = j
    for i in slaves:
        del eqs_to_add_dict[Symbol('%s'%i)]
        
    eqs_to_add_ready = copy.deepcopy(eqs_to_add_dict)    

#    for i in eqs_to_add_dict: #Changes s2 to (d/dt)s2
#        eqs_to_add_ready[Symbol('(d/dt)%s'%i)] = eqs_to_add_ready.pop(i)
    for l in eqs_to_add_ready.keys(): #Substitutes the values of the algebraic system
        for i, j in enumerate(sol_dict):
            eqs_to_add_ready[l]=eqs_to_add_ready[l].subs(sol_dict.keys()[i], sol_dict.values()[i])
    return eqs_to_add_ready 


def remove_minus_sign(expr):
    
    if expr.could_extract_minus_sign() == True:
       expr=expr*-1
   
    return expr   



def tropicalization(model):

    eqs_for_tropicalization = diff_alg_system(model) 
    tropicalized = {}
    borders = {}

#    for i in eqs_for_tropicalization.keys():
#        for par in model.parameters: eqs_for_tropicalization[i] = simplify(eqs_for_tropicalization[i].subs(par.name, par.value)) # Substitute parameters and simplify

    for j in eqs_for_tropicalization.keys():
        if type(eqs_for_tropicalization[j]) == Mul: print solve(log(j), dict = True) #If Mul=True there is only one monomial
        elif eqs_for_tropicalization[j] == 0: print 'there are not monomials'
        else:            
            ar = eqs_for_tropicalization[j].args #List of the terms of each equation
            asd=0 
            bor = []
            for l, k in enumerate(ar):
                p = k
                for f, h in enumerate(ar):
                   if k != h:
                      p *= Heaviside(log(remove_minus_sign(k)) - log(remove_minus_sign(h)))
                      bor.append(log(remove_minus_sign(k)) - log(remove_minus_sign(h)))
                borders[j] = bor  # this adds the arguments of the heaviside functions to the borders dict.    
                asd +=p
            tropicalized[j] = asd
    return borders, tropicalized    

def visualization(model):
    prueba = linspace(0, 100, 10001)
    eqs_to_graph = tropicalization(model)
    for l in sorted(eqs_to_graph.keys()):
        for i, j in enumerate(sorted(eqs_to_graph[l])):
            if j.has(Symbol('s6')) == False: print solve(j, Symbol('s4'), dict=True)              
            else:  print solve(j, Symbol('s6'), dict=True)
            plt.loglog(s4, )
            plt.show()   
    t = linspace(0.0001,1, 10001)
    t1 = linspace(1,1,10001)
    s = 0.00912870929175277*sqrt(1/t)

    plt.ion
    plt.xlabel('s4')
    plt.ylabel('s6')
    plt.loglog(t, s, color='b')
    plt.loglog(t,0.018*t,color='c') 
    plt.loglog(t, 0.00555555555555556/t, color= 'm')
    plt.loglog(t,0.0100000000000000*t1, color='k')
    plt.loglog(0.833333333333333*t1, t, color='g' )
    plt.show() 


#Numerical solution of the tropical differential equations

def diff_equa_to_solve(y, t):
    variables = []
    equations = []
    rhs_exprs = []
    ydot = [1,1,1]
    new_variables = {}
    
    tropical_system = tropicalization(model)[1]

    for i in tropical_system:
        variables.append(i) 

    for i, j in enumerate(variables):
        equations.append(tropical_system[j])
    new_variables[j] = 'y[%d]'%i

    for i in range(len(equations)):
       equations[i] = equations[i].subs('C0',1 )    

    for i, j in enumerate(variables):
            tempstring = re.sub(r's(\d+)', lambda m: new_variables[Symbol(((m.group())))], str(equations[i]))       
            rhs_exprs.append(compile(tempstring, '<ydot[%s]>' % i, 'eval'))  
    rhs_locals = {'y':y}
    for i in range(len(equations)):
        ydot[i] = eval(rhs_exprs[i], rhs_locals)
    
    return ydot

    variables_ready = []
    
    tropical_system = tropicalization(model)[1]
    
    for i in tropical_system:
       variables_ready.append(i)
       
    variables0=copy.deepcopy(variables_ready)
    
    #Initial conditions
    for i, j in enumerate(variables0):
       variables0[i]=variables0[i].subs(j, float(raw_input("Enter value of initial conditions of % s"%j  )))
       
    #Value of cycle constants
    
    #Ode solver parameters
    abserr = float(raw_input("Enter abserr parameter: "))
    relerr = float(raw_input("Enter relerr parameter: "))
    stoptime = float(raw_input("Enter stoptime parameter: "))
    numpoints = int(raw_input("Enter number of points parameter: "))
    
    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
    
    sol = odeint(diff_equa_to_solve, variables0, t)

tropicalization(model)

