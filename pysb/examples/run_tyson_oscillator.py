
# This creates model.odes which contains the math
from pysb.bng import generate_equations
from pysb.integrate import odesolve
from pysb.examples.tyson_oscillator import model
#from pylab import *
from numpy import *
import matplotlib.pyplot as plt

#ion()

generate_equations(model)

t = linspace(0, 100, 10001)
x = odesolve(model, t)

#plot(t, x['CT'])  # Good validation of mass balance for cdc2
#plot(t, x['YT']/x['CT'])
#plot(t, x['M']/x['CT'])
#show()

from sympy.solvers import solve
from sympy import Symbol
from sympy import symarray
from sympy.functions.elementary.complexes import Abs

# Right now, s1 is the only one not matching the paper. Needs further investigation, I think this 

# This function will take a model
# and return maximum distance between concrete species trace and imposed trace 
def find_slaves(model, t, ignore=15, epsilon=1e-6):
    slaves = []

    generate_equations(model)
    x = odesolve(model, t)
    x = x[ignore:] # Ignore first couple points
    t = t[ignore:]
    names = [n for n in filter(lambda n: n.startswith('__'), x.dtype.names)]
    x = x[names] # Only concrete species are considered
    names = [n.replace('__','') for n in names]
    x.dtype = [(n,'<f8') for n in names]

    for i, eq in enumerate(model.odes): # i is equation number
        eq   = eq.subs('s%d' % i, 's%dstar' % i)
        sol  = solve(eq, Symbol('s%dstar' % i)) # Find equation of imposed trace
        max  = None # Start with no distance between imposed traces and computed trace for this species
        for j in range(len(sol)):  # j is solution j for equation i
            prueba = zeros(len(x))
            for p in model.parameters: sol[j] = sol[j].subs(p.name, p.value) # Substitute parameters
            for l, tt in enumerate(t):
                prueba[l] = Abs(sol[j].evalf(subs={n:x[tt][n] for n in names}) - x[tt]['s%d'%i])            
            if (prueba.max() <= epsilon): slaves.append("s%d" % i)
#                if dist > max: max = dist
#        print("max[%d] = %lg" % (i, max))
#        distance.append(max)
        #if(max <= epsilon): slaves.append("s%d" % i) # Change to suit output as needed
    return slaves


zero = model.odes[0]
zero = zero.subs('s0', 's0stars')
zero = zero.subs('k6', model.parameters['k6'].value)
zero = zero.subs('k8', model.parameters['k8'].value)
zero = zero.subs('k9', model.parameters['k9'].value)

first = model.odes[1]
first = first.subs('s1', 's1stars')
first = first.subs('k1', model.parameters[0].value)
first = first.subs('k2', model.parameters[1].value)
first = first.subs('k3', model.parameters[2].value)

four = model.odes[4]
four = four.subs('s4', 's4stars')
four = four.subs('k8', model.parameters['k8'].value)
four = four.subs('k9', model.parameters['k9'].value)
four = four.subs('k3', model.parameters['k3'].value)

five = model.odes[5]
five = five.subs('s5', 's5stars')
five = five.subs('k4', model.parameters[3].value)
five = five.subs('kp4', model.parameters[4].value)
five = five.subs('k5', model.parameters[5].value)
five = five.subs('k3', model.parameters[6].value)


six = model.odes[6]
six = six.subs('s6', 's6stars')
six = six.subs('k4', model.parameters[3].value)
six = six.subs('kp4', model.parameters[4].value)
six = six.subs('k5', model.parameters[5].value)
six = six.subs('k6', model.parameters[6].value)


s0stars = Symbol("s0stars")
s4 = Symbol("s4")
s6 = Symbol("s6")

s1stars = Symbol("s1stars")
s2 = Symbol("s2")

s5stars = Symbol("s5stars")
s1 = Symbol("s1")

s6stars = Symbol("s6stars")
s5 = Symbol("s5")
s6 = Symbol("s6")

s4stars = Symbol("s4stars")
s0 = Symbol("s0")


y0 = solve(zero, s0stars)
y1 = solve(first, s1stars)
y4 = solve(four, s4stars)
y5 = solve(five, s5stars)
y6 = solve(six, s6stars)

B = symarray('s0', 10001)
C = symarray('s6', 10001)
D = symarray('s1', 10001)
F = symarray('s5', 10001)
G = symarray('s4', 10001)

for i in range(1, 10001):
    B[i] = y0[0].evalf(subs={s4: x['__s4'][i], s6: x['__s6'][i]})
    C[i] = y6[0].evalf(subs={s5: x['__s5'][i], s6: x['__s6'][i]})
    D[i] = y1[0].evalf(subs={s2: x['__s2'][i], s4: x['__s4'][i]})
    F[i] = y5[0].evalf(subs={s1: x['__s1'][i], s4: x['__s4'][i], s6: x['__s6'][i]})
    G[i] = y4[0].evalf(subs={s0: x['__s0'][i], s1: x['__s1'][i]})
   # print B[i]  

CAbs = [Abs(j) for j in C]     


l0star = plt.semilogy(t, B)
l0 = plt.semilogy(t, x['__s0'])
l1star = plt.semilogy(t, D)
l1 = plt.semilogy(t, x['__s1'])
l4star = plt.semilogy(t, G)
l4 = plt.semilogy(t, x['__s4'])
l5star = plt.semilogy(t, F)
l5 = plt.semilogy(t, x['__s5'])
l6star  = plt.semilogy(t, CAbs)
l6 = plt.semilogy(t, x['__s6'])
plt.show()
#plot(t, C)
   



