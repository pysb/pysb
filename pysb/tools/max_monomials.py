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
        self.mon_names          = {}
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
        if verbose: print "Getting maximum monomials data"
        self.drivers_max(self.y[ignore:], verbose)
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
    
    def drivers_max(self,y, verbose=False):
        drivers = list(set(range(len(self.model.species)))-set(self.passengers))
        drivers_data= {}
        drivers_monomials = {}
        for i in drivers:
            spe_monomials = self.model.odes[i].as_coefficients_dict()
            if spe_monomials.keys() == [1]: print "equation" + ' ' + str(i) + ' ' + 'does not have monomials, it is a constant'
            else:
                sign_monomials = self.model.odes[i].as_coefficients_dict().values()
                monomials_eval = numpy.zeros((len(spe_monomials), len(self.tspan[1:])),dtype=float)
                drivers_monomials[i] = spe_monomials
                tmp = numpy.zeros((len(spe_monomials), len(self.tspan[1:])), dtype=float)
                for q,j in enumerate(spe_monomials.keys()):
                    for par in self.model.parameters: j=j.subs(par.name,par.value)
                    arg_f1 = []
                    var_to_study = [atom for atom in j.atoms(sympy.Symbol) if not re.match(r'\d',str(atom))] #Variables of monomial 
                    f1 = sympy.lambdify(var_to_study, j, modules = dict(Heaviside=_Heaviside_num, log=numpy.log, Abs=numpy.abs)) 
                    for va in var_to_study:
                       arg_f1.append(y[str(va)])    
                    tmp[q]=f1(*arg_f1)
                for col in range(len(self.tspan[1:])):
                    mon_value = [(m,v) for m,v in enumerate(tmp[:,col]) if v >= 0.8*max(tmp[:,col])]
                    for va in mon_value:
                        monomials_eval[:,col][va[0]] = va[1]
                drivers_data[i] = monomials_eval
            self.tro_species = drivers_data
            self.mon_names = drivers_monomials
            
    def visualization(self, drivers_v = None):
        if drivers_v is not None:
            species_ready = []
            for i in drivers_v:
                if i in self.tro_species.keys(): species_ready.append(i)
                else: print 'specie' + ' ' + str(i) + ' ' + 'is not a driver'
        elif driver_species is None:
            raise Exception('list of driver species must be defined')
        
        if species_ready == []:
            raise Exception('None of the input species is a driver')
             
            
        spe_index = {}
        for i, j in enumerate(self.model.species):
            spe_index[i] = '__s%d'%i
        
        
        
        colors = itertools.cycle(["b", "g", "c", "m", "y", "k" ])
        
        for sp in species_ready:
            si_flux = 0
            no_flux = 0
            monomials_dic = self.tro_species[sp]
            f = plt.figure()
            plt.subplot(211)
            monomials = []
            monomials_inf = self.mon_names[sp]
            for m_value, name in zip(monomials_dic,monomials_inf.keys()):
                x_concentration = numpy.nonzero(m_value)[0]
                if len(x_concentration) > 0:   
                    monomials.append(name)            
                    si_flux+=1
                    x_points = [self.tspan[x] for x in x_concentration] 
                    prueba_y = numpy.repeat(2*si_flux, len(x_concentration))
                    if monomials_inf[name]== 1 : plt.scatter(x_points[::int(len(self.tspan)/100)], prueba_y[::int(len(self.tspan)/100)], color = next(colors), marker=r'$\uparrow$', s=numpy.array([m_value[k] for k in x_concentration])[::int(len(self.tspan)/100)]*2)
                    if monomials_inf[name]==-1 : plt.scatter(x_points[::int(len(self.tspan)/100)], prueba_y[::int(len(self.tspan)/100)], color = next(colors), marker=r'$\downarrow$', s=numpy.array([m_value[k] for k in x_concentration])[::int(len(self.tspan)/100)]*2)
                else: no_flux+=1
            y_pos = numpy.arange(2,2*si_flux+4,2)    
            plt.yticks(y_pos, monomials, size = 'x-small') 
            plt.ylabel('Monomials')
            plt.xlim(0, self.tspan[-1])
            plt.subplot(210)
            plt.plot(self.tspan[1:],self.y['__s%d'%sp][1:])
            plt.ylabel('Molecules')
            plt.xlabel('Time (s)')
            plt.suptitle('Tropicalization' + ' ' + str(self.model.species[sp]))
        return f 
       
    def get_driver_data(self):
        return self.tro_species

def run_tropical(model, tspan, parameters = None, sp_visualize = None):
    tr = Tropical(model)
    tr.tropicalize(tspan, parameters)
    if sp_visualize is not None:
        tr.visualization(drivers_v=sp_visualize)
    return tr.get_driver_data()

