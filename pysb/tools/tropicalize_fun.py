from __future__ import division
import networkx
import sympy
import re
import copy
import numpy
import sympy.parsing.sympy_parser
import itertools
import matplotlib.pyplot as plt
import pysb
import csv
from pysb.integrate import odesolve
from matplotlib.font_manager import FontProperties



def parse_name(spec):
    m = spec.monomer_patterns
    lis_m = []
    for i in range(len(m)):
        tmp_1 = str(m[i]).partition('(')
        tmp_2 = re.findall(r"(?<=\').+(?=\')",str(m[i]))
        lis_m.append(''.join([tmp_1[0],tmp_2[0]]))
    return '_'.join(lis_m)
                

def rescale(values, new_min = 10, new_max = 200):
    output = []
    old_min, old_max = min(values), max(values)

    for v in values:
        new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
        output.append(new_v)

    return output


def Heaviside_num(x):
    return 0.5*(numpy.sign(x)+1)

def y_ready(model,t, ignore=1, pars = None):
    y = odesolve(model, t, pars)
    
    # Only concrete species are considered, and the names must be made to match
    names           = [n for n in filter(lambda n: n.startswith('__'), y.dtype.names)]
    y          = y[names]
    y.dtype    = [(n,'<f8') for n in names]    
    return y[ignore:]
  


def find_slaves(model, t, epsilon=2, p=None):
    y = y_ready(model,t, pars=p)
    slaves = []
    a = []               # list of solved polynomial equations
    b = []               # b is the list of differential equations   

    # Loop through all equations (i is equation number)
    for i, eq in enumerate(model.odes):
        eq        = eq.subs('__s%d' % i, '__s%dstar' % i)
        sol       = sympy.solve(eq, sympy.Symbol('__s%dstar' % i))        # Find equation of imposed trace
        for j in range(len(sol)):        # j is solution j for equation i (mostly likely never greater than 2)
            for p in model.parameters: sol[j] = sol[j].subs(p.name, p.value)    # Substitute parameters
            a.append(sol[j])
            b.append(i)
    for k,e in enumerate(a):    # a is the list of solution of polinomial equations, b is the list of differential equations
        args = []               #arguments to put in the lambdify function
        variables = [atom for atom in a[k].atoms(sympy.Symbol) if not re.match(r'\d',str(atom))]
        f = sympy.lambdify(variables, a[k], modules = dict(sqrt=numpy.lib.scimath.sqrt) )
        for u,l in enumerate(variables):
            args.append(y[:][str(l)])
        hey = abs(f(*args) - y[:]['__s%d'%b[k]])
        s_points = sum(w < epsilon for w in hey)
        if s_points > 0.9*len(hey) : slaves.append(b[k])
#         if hey.max() <= epsilon : slaves.append(b[k])            
            
    return slaves


# This is a function which builds the edges according to the nodes
def r_link(graph, s, r, **attrs):
    nodes = (s, r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = reversed(nodes)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)

def construct_graph(model):
    if(model.odes == None or model.odes == []):
        pysb.bng.generate_equations(model)

    graph = networkx.DiGraph(rankdir="LR")
    ic_species = [cp for cp, parameter in model.initial_conditions]
    for i, cp in enumerate(model.species):
        species_node = i
        graph.add_node(species_node, label=species_node)
    for i, reaction in enumerate(model.reactions):       
        reactants = set(reaction['reactants'])
        products = set(reaction['products']) 
        attr_reversible = {}
        for s in reactants:
            for p in products:
                r_link(graph, s, p, **attr_reversible)
    return graph

#This function finds conservation laws from the conserved cycles
def mass_conserved(model, verbose=False, p=None):
    if(model.odes == None or model.odes == []):
        pysb.bng.generate_equations(model)
    y = y_ready(model,t, pars=p)    
    h = [] # Array to hold conservation equation
    g = [] # Array to hold corresponding lists of free variables in conservation equations
    value_constants = {} #Dictionary that storage the value of each constant
    cycles = list(networkx.simple_cycles(construct_graph(model)))
    for i, item in enumerate(cycles):
        b = 0
        u = 0
        for j, specie in enumerate(item):
            b += model.odes[cycles[i][j]]
        if b == 0:
            g.append(item)
            for l,k in enumerate(item):
                u += sympy.Symbol('__s%d' % cycles[i][l])    
            h.append(u-sympy.Symbol('C%d'%i))
            if verbose: print '  cycle%d'%i, 'is conserved'
    
    for i in h:
        constant_to_solve = [atom for atom in i.atoms(sympy.Symbol) if re.match(r'[C]',str(atom))]
        solution = sympy.solve(i, constant_to_solve ,implicit = True)
        solution_ready = solution[0]
        for q in solution_ready.atoms(sympy.Symbol): solution_ready = solution_ready.subs(q, y[0][str(q)])
        value_constants[constant_to_solve[0]] = solution_ready
        
    (conservation, conserve_var, value_conservation) = h, g, value_constants     
    return h, g, value_constants

def slave_equations(model, p1=None):
    if(model.odes == None or model.odes == []):
        eq = model.odes
    slave_conserved_eqs = {}
    slaves = find_slaves(model, t, p=p1)
    for i, j in enumerate(slaves):
        slave_conserved_eqs[j] = model.odes[slaves[i]]
    return slave_conserved_eqs

def find_nearest_zero(array):
    idx = (numpy.abs(array)).argmin()
    return array[idx]

# Make sure this is the "ignore:" y
def pruned_equations(model, t, rho=1, p2=None):
    pruned_eqs = slave_equations(model, p1=p2)
    eqs        = copy.deepcopy(pruned_eqs)
    conservation = mass_conserved(model, p=p2)[0]
    y = y_ready(model,t, pars=p2)
    
    for i, j in enumerate(eqs):
        ble = eqs[j].as_coefficients_dict().keys() # Get monomials
        for l, m in enumerate(ble): #Compares the monomials to find the pruned system
            m_ready = m # Monomial to compute with
            m_elim  = m # Monomial to save
            for p in model.parameters: m_ready = m_ready.subs(p.name, p.value) # Substitute parameters
            for k in range(len(ble)):
                if (k+l+1) <= (len(ble)-1):
                    ble_ready = ble[k+l+1] # Monomial to compute with
                    ble_elim  = ble[k+l+1] # Monomial to save
                    for p in model.parameters: ble_ready = ble_ready.subs(p.name, p.value) # Substitute parameters
                    args2 = []
                    args1 = []
                    variables_ble_ready = [atom for atom in ble_ready.atoms(sympy.Symbol) if not re.match(r'\d',str(atom))]
                    variables_m_ready = [atom for atom in m_ready.atoms(sympy.Symbol) if not re.match(r'\d',str(atom))]
                    f_ble = sympy.lambdify(variables_ble_ready, ble_ready, 'numpy' )
                    f_m = sympy.lambdify(variables_m_ready, m_ready, 'numpy' )
                    for uu,ll in enumerate(variables_ble_ready):
                        args2.append(y[:][str(ll)])
                    for w,s in enumerate(variables_m_ready):
                        args1.append(y[:][str(s)])
                    hey_pruned = f_m(*args1) - f_ble(*args2)
                    diff = find_nearest_zero(hey_pruned)
                    diff_pru = numpy.abs(diff)
                    if diff > 0 and diff_pru > rho:
                        pruned_eqs[j] = pruned_eqs[j].subs(ble_elim, 0)
                    if diff < 0 and diff_pru > rho:\
                        pruned_eqs[j] = pruned_eqs[j].subs(m_elim, 0)   
                        
    for i, l in enumerate(conservation): #Add the conservation laws to the pruned system
        pruned_eqs['cons%d'%i]=l

    return pruned_eqs

def solve_pruned(model,t, p3=None):
    solve_for = copy.deepcopy(find_slaves(model, t, p=p3))
    eqs       = copy.deepcopy(pruned_equations(model, t, p2=p3))
    eqs_l = []
    conserve_var = mass_conserved(model,p=p3)[1]
    for i in eqs.keys():
        eqs_l.append(eqs[i])
        
    
    for i in conserve_var:
        if len(i) == 1:
            solve_for.append(i[0])
    variables =  [sympy.Symbol('__s%d' %var) for var in solve_for ]
    sol = sympy.solve(eqs_l, variables)

    if len(sol) == 0:
        sol_pruned = { j:sympy.Symbol('__s%d'%j) for i, j in enumerate(solve_for) }
    else:
        sol_pruned = { j:sol[0][i] for i, j in enumerate(solve_for) }
   
    return sol_pruned

def equations_to_tropicalize(model,t, p4=None):
    idx = list(set(range(len(model.odes))) - set(solve_pruned(model,t,p3=p4).keys()))
    eqs = { i:model.odes[i] for i in idx }
    value_conservation = mass_conserved(model, p=p4)[2]
    for l in eqs.keys(): #Substitutes the values of the algebraic system
#             for k in self.sol_pruned.keys(): eqs[l]=eqs[l].subs(sympy.Symbol('s%d' % k), self.sol_pruned[k])
        for q in value_conservation.keys(): eqs[l] = eqs[l].subs(q, value_conservation[q])
#         for i in eqs.keys():
#             for par in self.model.parameters: eqs[i] = sympy.simplify(eqs[i].subs(par.name, par.value))
    return eqs

def final_tropicalization(model,t,p5=None):
    eqs_for_tropicalization = equations_to_tropicalize(model,t, p4=p5)
    tropicalized = {}
    
    for j in sorted(eqs_for_tropicalization.keys()):
        if type(eqs_for_tropicalization[j]) == sympy.Mul: tropicalized[j] = eqs_for_tropicalization[j]  #If Mul=True there is only one monomial
        elif eqs_for_tropicalization[j] == 0: print 'there are no monomials'
        else:            
            ar = eqs_for_tropicalization[j].args #List of the terms of each equation  
            asd=0 
            for l, k in enumerate(ar):
                p = k
                for f, h in enumerate(ar):
                   if k != h:
                      p *= sympy.Heaviside(sympy.log(abs(k)) - sympy.log(abs(h)))
                asd +=p
            tropicalized[j] = asd

    return tropicalized


def range_dominating_monomials(model, t, p6=None, nam=None): 
    pysb.bng.generate_equations(model)
    y = y_ready(model,t,pars=p6)
    
    spe_name = {}
    for i, j in enumerate(model.species):
        spe_name['__s%d'%i] = j
    
    tropical_system = final_tropicalization(model,t,p5=p6)
       
    obs_all = []
    for i in model.observables:
        obs_spe = []
        tropical_obs = {}
        for j in i.species:
            obs_spe.append(j)
        obs_in_trop = list(set(tropical_system)&set(obs_spe))   
        for i in obs_in_trop:
            tropical_obs[i]=tropical_system[i]
        obs_all.append(tropical_obs)
    
    colors = itertools.cycle(["b", "g", "c", "m", "y", "k" ])
    
    

    for ii, obs in enumerate(obs_all):
        monomials = []
        vertical = []
        all_variables = []
        dom_variables = []
        count = 0
        plt.figure(1)
        plt.subplot(210)
        for i in obs.keys():
            no_flux = 0
            si_flux = 0
            mons = obs[i].as_coefficients_dict().keys()
            sign_monomial = obs[i].as_coefficients_dict().values()
            mols_time = numpy.zeros(len(t)-1)
            for j,q in zip(mons,sign_monomial):
                jj = copy.deepcopy(j)
                for par in model.parameters: j=j.subs(par.name,par.value)
                arg_f1 = []
                var_to_study = [atom for atom in j.atoms(sympy.Symbol) if not re.match(r'\d',str(atom))] #Variables of monomial 
                all_variables.append(var_to_study)
                f1 = sympy.lambdify(var_to_study, j, modules = dict(Heaviside=Heaviside_num, log=numpy.log, Abs=numpy.abs)) 
                for va in var_to_study:
                   arg_f1.append(y[str(va)])    
                x_concentration = numpy.nonzero(f1(*arg_f1))[0].tolist() # Gives the positions of nonzero numbers
                if len(x_concentration) > 0:
                    monomials.append(str(jj).partition('*Heaviside')[0])  
                    tmp = sympy.parsing.sympy_parser.parse_expr(str(j).partition('*Heaviside')[0])
                    dom_variables.append([atom for atom in tmp.atoms(sympy.Symbol) if not re.match(r'\d',str(atom))])
                    si_flux+=1
                else: no_flux+=1
                y_pos = numpy.arange(2,2*len(monomials)+4,2)
                count = 2*len(monomials)
                if len(x_concentration) > 0: vertical.append(x_concentration[-1])
                for ij in range(len(x_concentration)-1):
                    if x_concentration[ij] == x_concentration[ij+1]-1:
                       pass
                    else: vertical.append(x_concentration[ij])
                mols_time = mols_time + f1(*arg_f1)
                x_points = [t[x] for x in x_concentration] 
                prueba_y = numpy.repeat(count, len(x_points))
                pru = f1(*arg_f1)
                if q==1 : plt.scatter(x_points[::120], prueba_y[::120], color = next(colors), marker=r'$\uparrow$', s=numpy.array([pru[k] for k in x_concentration])[::120]*0.4)
                if q==-1 : plt.scatter(x_points[::120], prueba_y[::120], color = next(colors), marker=r'$\downarrow$', s=numpy.array([pru[k] for k in x_concentration])[::120]*0.4)
                plt.xlim(0, t[len(t)-1])
                plt.ylim(0, len(mons)+1) 
                
            plt.yticks(y_pos, monomials, size = 'x-small') 
            plt.xlabel('Time (s)')
            plt.ylabel('Monomials')
            print str(model.species[i]) + ' ' +'Monomials that do not contribute to flux:' +' '+ str(no_flux) +', '+ 'Monomias that contribute to flux' +' '+ str(si_flux)
            
        smac_activated = y['__s67']+y['__s72']
        max_smac = max(smac_activated)
        smac_90 = min(smac_activated, key=lambda x:abs(x-max_smac*0.9)) 
        smac_10 = min(smac_activated, key=lambda x:abs(x-max_smac*0.1))   
        death = (smac_90+smac_10)/2
        smac_death = min(smac_activated, key=lambda x:abs(x-death)) 
        t_death = [ik for ik,xy in enumerate(smac_activated) if xy == smac_death]
        plt.vlines(t_death[0],0,y_pos[-1], color='b', linestyle=':' )
        
        plt.subplot(211)
        for l in range(len(obs_all)):
            for sp in obs_all[l].keys():
                plt.plot(t[1:], y['__s%d'%sp], label=parse_name(model.species[sp]))
                plt.legend(loc=2,prop={'size':6})
        plt.title('Tropicalization' + ' ' + str(model.observables[ii].reaction_pattern) )
        plt.ylabel('Molecules')
        plt.ylim(0,20000)
        plt.show()
#         plt.savefig('/home/carlos/Desktop/test_parameters_embedded/'+nam+'__s%d'%i, bbox_inches='tight', dpi=800, format='pdf')    
        plt.close(plt.figure(1))
    return
                     

from earm.lopez_embedded import model
t= numpy.linspace(0, 20000, 20001)          # timerange used
# from pysb.examples.tyson_oscillator import model
# t= numpy.linspace(0, 100, 100) 

f = open('/home/carlos/Desktop/pars_embedded.txt') 
data = csv.reader(f)
params = []
for i in data:params.append(float(i[1]))
range_dominating_monomials(model, t,p6=params,nam='')
