import gillespy
import re
import sympy
from pysb import bng

    
class pysb2gill(gillespy.Model):
    def __init__(self, parameter_values=None):
        bng.generate_equations(pysb_model)
        system_volume=check_for_volumne(pysb_model)
        gillespy.Model.__init__(self, name=pysb_model.name,volume=system_volume)
        convert_parameters(self,pysb_model)
        convert_species(self,pysb_model)
        convert_reactions(self,pysb_model)


def convert(Pysb_model):
    global pysb_model
    pysb_model = Pysb_model
    gill_model = pysb2gill()
    return gill_model



def check_for_volumne(model):
    for i in xrange(len(model.parameters)):
        if str(model.parameters[i]) == 'VOL':
            for j in model.parameters_unused():
                if str(model.parameters[i]) == str(j):
                    return 1.0
            print "Volume found in parameters", model.parameters[i].value
            return model.parameters[i].value
        else:
            return 1.0


def convert_parameters(self,model):
    param_list = []
    for i in xrange(len(model.parameters)):
        found = False
        for j in model.parameters_unused():
            if str(model.parameters[i]) == str(j):
                #print model.parameters[i], ' unused'
                found = True
        if found == False:
            print model.parameters[i].name,model.parameters[i].value
            param_list.append(gillespy.Parameter(name=model.parameters[i].name\
                                             , expression=model.parameters[i].value))
    self.add_parameter(param_list)

def convert_species(self,model):
    species_list = []
    for i in xrange(len(model.species)):
        #print model.species[i]
        found = False
        for j in xrange(len(model.initial_conditions)):
            #print model.initial_conditions[j][0],model.species[i]
            if str(model.initial_conditions[j][0]) == str(model.species[i]):
                #print model.initial_conditions[j][1].value
                conc = model.initial_conditions[j][1].value*check_for_volumne(model)
                print model.species[i],"Population ",int(conc)
                species_list.append(gillespy.Species(name = "__s%d"%i , \
                         initial_value=int(conc)))
                found = True
        if found == False:
            #print model.species[i]," = 0"
            species_list.append(gillespy.Species(name = "__s%d"%i , initial_value=0))
    
    self.add_species(species_list)
    
def convert_reactions(self,model):
    reactions_list = []
    counter = 1
    for each in model.reactions:
        #print each
        reactants = dict()
        products = dict()
        # Forward           
        for react in each["reactants"]:
            react = "__s%d" % react
            if react in reactants:
                reactants[react] += 1
            else:
                reactants[react] = 1
        for prod in each["products"]:
            prod = "__s%d" % prod
            if prod in products:
                products[prod] += 1
            else:
                products[prod] = 1
        code = sympy.fcode(each["rate"])
        code = code.replace('**', '^')
        rate = str(code)
        for e in model.expressions:
            rate = re.sub(r'\b%s\b' % e.name, '('+sympy.ccode(e.expand_expr())+')', rate)
        for obs in model.observables:
            #print obs
            obs_string = ''
            for i in range(len(obs.coefficients)):
                if i > 0: obs_string += "+"
                if obs.coefficients[i] > 1: obs_string += str(obs.coefficients[i])+"*"
                obs_string += "__s"+str(obs.species[i])
            if len(obs.coefficients) > 1: obs_string = '(' + obs_string + ')'
            
            matches = re.findall('(s\d+)\^(\d+)', rate)
            for m in matches:
                repl = m[0]
                for i in range(1,int(m[1])):
                    repl += "*__%s" % m[0]
                rate = re.sub('s\d+\^\d+', repl, rate, count=1)
            rate = re.sub(r'%s' % obs.name, obs_string, rate)            
        print rate
        tmp = gillespy.Reaction(name = str(each["rule"])+"__%d"%counter,\
                                                reactants = reactants,\
                                                products = products,\
                                                propensity_function = rate)
        
        reactions_list.append(tmp)
        counter+=1
    self.add_reaction(reactions_list)
        
