
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


zero = model.odes[0]
zero = zero.subs('s0', 's0stars')
zero = zero.subs('k6', model.parameters[6].value)
zero = zero.subs('k8', model.parameters[8].value)
zero = zero.subs('k9', model.parameters[9].value)

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

for i in range(0, 10001):
  
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
   


