import networkx
import sympy
import re
import copy
import numpy
import sympy.parsing.sympy_parser
import itertools
import matplotlib.pyplot as plt
import pysb
from pysb.integrate import odesolve


def _Heaviside_num(x):
    return 0.5*(numpy.sign(x)+1)

def _parse_name(spec):
    m = spec.monomer_patterns
    lis_m = []
    for i in range(len(m)):
        tmp_1 = str(m[i]).partition('(')
        tmp_2 = re.findall(r"(?<=\').+(?=\')",str(m[i]))
        if tmp_2 == []: lis_m.append(tmp_1[0])
        else:
            lis_m.append(''.join([tmp_1[0],tmp_2[0]]))
    return '_'.join(lis_m)

class Tropical:
    def __init__(self, model):
        self.model              = model
        self.tspan              = None
        self.y                  = None  # ode solution, numpy array
        self.passengers         = None
        self.graph              = None
        self.cycles             = []
        self.conservation       = None
        self.conserve_var       = None
        self.value_conservation = {}
        self.tro_species        = {}

    def __repr__(self):
        return "<%s '%s' (passengers: %s, cycles: %d) at 0x%x>" % \
            (self.__class__.__name__, self.model.name,
             self.passengers.__repr__(),
             len(self.cycles),
             id(self))

    def tropicalize(self,tspan=None, param_values=None, ignore=1, epsilon=2, rho=3, verbose=True):
        if verbose: print "Solving Simulation"
        
        if tspan is not None:
            self.tspan = tspan
        elif self.tspan is None:
            raise Exception("'tspan' must be defined.")
        
        self.y = odesolve(self.model, self.tspan, param_values)
        
        # Only concrete species are considered, and the names must be made to match
#         names           = [n for n in filter(lambda n: n.startswith('__'), self.y.dtype.names)]
#         self.y          = self.y[names]
#         self.y.dtype    = [(n,'<f8') for n in names]    
          
        if verbose: print "Getting Passenger species"
        self.find_passengers(self.y[ignore:], verbose, epsilon)
        if verbose: print "Constructing Graph"
        self.construct_graph()
        if verbose: print "Computing Cycles"
        self.cycles = list(networkx.simple_cycles(self.graph))
        if verbose: print "Computing Conservation laws"
        (self.conservation, self.conserve_var, self.value_conservation) = self.mass_conserved(self.y[ignore:])
        if verbose: print "Pruning Equations"
        self.pruned = self.pruned_equations(self.y[ignore:], rho)
        if verbose: print "Solving pruned equations"
        self.sol_pruned = self.solve_pruned()
        if verbose: print "equation to tropicalize"
        self.eqs_for_tropicalization = self.equations_to_tropicalize()
        if verbose: print "Getting tropicalized equations"
        self.tropical_eqs = self.final_tropicalization()
        self.data_drivers(self.y[ignore:])
        
        return 

    def find_passengers(self, y, verbose=False, epsilon=None):
        self.passengers = []
        a = []               # list of solved polynomial equations
        b = []               # b is the list of differential equations   

        # Loop through all equations (i is equation number)
        for i, eq in enumerate(self.model.odes):
            eq        = eq.subs('__s%d' % i, '__s%dstar' % i)
            sol       = sympy.solve(eq, sympy.Symbol('__s%dstar' % i))        # Find equation of imposed trace
            for j in range(len(sol)):        # j is solution j for equation i (mostly likely never greater than 2)
                for p in self.model.parameters: sol[j] = sol[j].subs(p.name, p.value)    # Substitute parameters
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
            if s_points > 0.9*len(hey) : self.passengers.append(b[k])
                
        return self.passengers


    # This is a function which builds the edges according to the nodes
    def r_link(self, graph, s, r, **attrs):
        nodes = (s, r)
        if attrs.get('_flip'):
            del attrs['_flip']
            nodes = reversed(nodes)
        attrs.setdefault('arrowhead', 'normal')
        graph.add_edge(*nodes, **attrs)

    def construct_graph(self):
        if(self.model.odes == None or self.model.odes == []):
            pysb.bng.generate_equations(model)

        self.graph = networkx.DiGraph(rankdir="LR")
        ic_species = [cp for cp, parameter in self.model.initial_conditions]
        for i, cp in enumerate(self.model.species):
            species_node = i
            self.graph.add_node(species_node, label=species_node)
        for i, reaction in enumerate(self.model.reactions):       
            reactants = set(reaction['reactants'])
            products = set(reaction['products']) 
            attr_reversible = {}
            for s in reactants:
                for p in products:
                    self.r_link(self.graph, s, p, **attr_reversible)
        return self.graph

    #This function finds conservation laws from the conserved cycles
    def mass_conserved(self, y, verbose=False):
        if(self.model.odes == None or self.model.odes == []):
            pysb.bng.generate_equations(self.model)
        h = [] # Array to hold conservation equation
        g = [] # Array to hold corresponding lists of free variables in conservation equations
        value_constants = {} #Dictionary that storage the value of each constant
        for i, item in enumerate(self.cycles):
            b = 0
            u = 0
            for j, specie in enumerate(item):
                b += self.model.odes[self.cycles[i][j]]
            if b == 0:
                g.append(item)
                for l,k in enumerate(item):
                    u += sympy.Symbol('__s%d' % self.cycles[i][l])    
                h.append(u-sympy.Symbol('C%d'%i))
                if verbose: print '  cycle%d'%i, 'is conserved'
        
        for i in h:
            constant_to_solve = [atom for atom in i.atoms(sympy.Symbol) if re.match(r'[C]',str(atom))]
            solution = sympy.solve(i, constant_to_solve ,implicit = True)
            solution_ready = solution[0]
            for q in solution_ready.atoms(sympy.Symbol): solution_ready = solution_ready.subs(q, y[0][str(q)])
            value_constants[constant_to_solve[0]] = solution_ready
            
        (self.conservation, self.conserve_var, self.value_conservation) = h, g, value_constants     
        return h, g, value_constants

    def passenger_equations(self):
        if(self.model.odes == None or self.model.odes == []):
            pysb.bng.generate_equations(self.model)
            eq = self.model.odes
        passenger_conserved_eqs = {}
        for i, j in enumerate(self.passengers):
            passenger_conserved_eqs[j] = self.model.odes[self.passengers[i]]
        return passenger_conserved_eqs

    def find_nearest_zero(self, array):
        idx = (numpy.abs(array)).argmin()
        return array[idx]

    # Make sure this is the "ignore:" y
    def pruned_equations(self, y, rho=1):
        pruned_eqs = self.passenger_equations()
        eqs        = copy.deepcopy(pruned_eqs)

        for i, j in enumerate(eqs):
            ble = eqs[j].as_coefficients_dict().keys() # Get monomials
            for l, m in enumerate(ble): #Compares the monomials to find the pruned system
                m_ready = m # Monomial to compute with
                m_elim  = m # Monomial to save
                for p in self.model.parameters: m_ready = m_ready.subs(p.name, p.value) # Substitute parameters
                for k in range(len(ble)):
                    if (k+l+1) <= (len(ble)-1):
                        ble_ready = ble[k+l+1] # Monomial to compute with
                        ble_elim  = ble[k+l+1] # Monomial to save
                        for p in self.model.parameters: ble_ready = ble_ready.subs(p.name, p.value) # Substitute parameters
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
                        diff = self.find_nearest_zero(hey_pruned)
                        diff_pru = numpy.abs(diff)
                        if diff > 0 and diff_pru > rho:
                            pruned_eqs[j] = pruned_eqs[j].subs(ble_elim, 0)
                        if diff < 0 and diff_pru > rho:\
                            pruned_eqs[j] = pruned_eqs[j].subs(m_elim, 0)   
                            
        for i, l in enumerate(self.conservation): #Add the conservation laws to the pruned system
            pruned_eqs['cons%d'%i]=l
        self.pruned = pruned_eqs
        return pruned_eqs

    def solve_pruned(self):
        solve_for = copy.deepcopy(self.passengers)
        eqs       = copy.deepcopy(self.pruned)
        eqs_l = []
        for i in eqs.keys():
            eqs_l.append(eqs[i])
            
        
        for i in self.conserve_var:
            if len(i) == 1:
                solve_for.append(i[0])
        variables =  [sympy.Symbol('__s%d' %var) for var in solve_for ]
        sol = sympy.solve(eqs_l, variables)

        if len(sol) == 0:
            self.sol_pruned = { j:sympy.Symbol('__s%d'%j) for i, j in enumerate(solve_for) }
        else:
            self.sol_pruned = { j:sol[0][i] for i, j in enumerate(solve_for) }
       
        return self.sol_pruned

    def equations_to_tropicalize(self):
        idx = list(set(range(len(self.model.odes))) - set(self.sol_pruned.keys()))
        eqs = { i:self.model.odes[i] for i in idx }

        for l in eqs.keys(): #Substitutes the values of the algebraic system
#             for k in self.sol_pruned.keys(): eqs[l]=eqs[l].subs(sympy.Symbol('s%d' % k), self.sol_pruned[k])
            for q in self.value_conservation.keys(): eqs[l] = eqs[l].subs(q, self.value_conservation[q])
#         for i in eqs.keys():
#             for par in self.model.parameters: eqs[i] = sympy.simplify(eqs[i].subs(par.name, par.value))
        self.eqs_for_tropicalization = eqs

        return eqs
    
    def final_tropicalization(self):
        tropicalized = {}
        
        for j in sorted(self.eqs_for_tropicalization.keys()):
            if type(self.eqs_for_tropicalization[j]) == sympy.Mul: tropicalized[j] = self.eqs_for_tropicalization[j] #If Mul=True there is only one monomial
            elif self.eqs_for_tropicalization[j] == 0: print 'there are no monomials'
            else:            
                ar = self.eqs_for_tropicalization[j].args #List of the terms of each equation  
                asd=0 
                for l, k in enumerate(ar):
                    p = k
                    for f, h in enumerate(ar):
                       if k != h:
                          p *= sympy.Heaviside(sympy.log(abs(k)) - sympy.log(abs(h)))
                    asd +=p
                tropicalized[j] = asd

        self.tropical_eqs = tropicalized
        return tropicalized


    def data_drivers(self, y):        
        colors = itertools.cycle(["b", "g", "c", "m", "y", "k" ])
        tropical_system = self.final_tropicalization()
        trop_data = {}

        for i in tropical_system.keys():
            mons_data = {}
            mons = tropical_system[i].as_coefficients_dict().keys()
            sign_monomial = tropical_system[i].as_coefficients_dict().values()
            for j,q in zip(mons,sign_monomial):
                test = [None]*2
                jj = copy.deepcopy(j) 
                for par in self.model.parameters: j=j.subs(par.name,par.value)
                arg_f1 = []
                var_to_study = [atom for atom in j.atoms(sympy.Symbol) if not re.match(r'\d',str(atom))] #Variables of monomial 
                f1 = sympy.lambdify(var_to_study, j, modules = dict(Heaviside=_Heaviside_num, log=numpy.log, Abs=numpy.abs)) 
                for va in var_to_study:
                   arg_f1.append(y[str(va)])    
                test[0]=f1(*arg_f1)
                test[1]=q
                mons_data[str(jj).partition('*Heaviside')[0]] = test
            trop_data[str(self.model.species[i])] = mons_data
        self.tro_species = trop_data
        return trop_data 
    
    def visualization(self, driver_species=None):
        if driver_species is not None:
            species_ready = []
            for i in driver_species:
                if i in self.tro_species.keys(): species_ready.append(i)
                else: print 'specie' + ' ' + str(i) + ' ' + 'is not a driver'
            species_ready = [i for i in driver_species if i in self.tro_species.keys()]
        elif driver_species is None:
            raise Exception('list of driver species must be defined')
        
        if species_ready == []:
            raise Exception('None of the input species is a driver')
             
            
        spe_index = {}
        for i, j in enumerate(self.model.species):
            spe_index[str(j)] = '__s%d'%i
        
        
        
        colors = itertools.cycle(["b", "g", "c", "m", "y", "k" ])
        
        for sp in species_ready:
            si_flux = 0
            no_flux = 0
            monomials_dic = self.tro_species[str(sp)]
            f = plt.figure()
            plt.subplot(211)
            monomials = []
            for c, mon in enumerate(monomials_dic):
                x_concentration = numpy.nonzero(monomials_dic[mon][0])[0]
                if len(x_concentration) > 0:   
                    monomials.append(mon)            
                    si_flux+=1
                    x_points = [self.tspan[x] for x in x_concentration] 
                    prueba_y = numpy.repeat(2*si_flux, len(x_concentration))
                    if monomials_dic[mon][1]==1 : plt.scatter(x_points[::int(len(self.tspan)/100)], prueba_y[::int(len(self.tspan)/100)], color = next(colors), marker=r'$\uparrow$', s=numpy.array([monomials_dic[mon][0][k] for k in x_concentration])[::int(len(self.tspan)/100)]*2)
                    if monomials_dic[mon][1]==-1 : plt.scatter(x_points[::int(len(self.tspan)/100)], prueba_y[::int(len(self.tspan)/100)], color = next(colors), marker=r'$\downarrow$', s=numpy.array([monomials_dic[mon][0][k] for k in x_concentration])[::int(len(self.tspan)/100)]*2)
                else: no_flux+=1
            y_pos = numpy.arange(2,2*si_flux+4,2)    
            plt.yticks(y_pos, monomials, size = 'x-small') 
            plt.ylabel('Monomials')
            plt.xlim(0, self.tspan[-1])
            plt.subplot(210)
            plt.plot(self.tspan[1:],self.y[spe_index[sp]][1:])
            plt.ylabel('Molecules')
            plt.xlabel('Time (s)')
            plt.suptitle('Tropicalization' + ' ' + sp)
            plt.show()
        return f  

    def get_trop_data(self):
        return self.tro_species
    def get_passengers(self):
        return self.passengers
    def get_drivers(self):
        return self.tro_species.keys()

def run_tropical(model, tspan, parameters = None, sp_visualize = None):
    tr = Tropical(model)
    tr.tropicalize(tspan, parameters)
    if sp_visualize is not None:
        tr.visualization(driver_species=sp_visualize)
    return tr.get_trop_data()