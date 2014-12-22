import numpy
import sympy
import matplotlib.pyplot as plt

y = (sympy.Symbol('x')**sympy.Symbol('n'))/((sympy.Symbol('k')**sympy.Symbol('n'))+(sympy.Symbol('x')**sympy.Symbol('n')))
y=y.subs(sympy.Symbol('x'), 0.1)
y=y.subs(sympy.Symbol('k'), 0.2)
f = sympy.lambdify([sympy.Symbol('n')], y, 'numpy' )
n_values = numpy.arange(0,100,1)
ss = n_values.reshape(1,100)
plt.subplot(211)
plt.plot(n_values, f(*ss))
plt.xlabel('n')
plt.ylabel('Fraction of y')
plt.title('variable parameter: n')
plt.subplot(212)
plt.loglog(n_values, f(*ss))
plt.xlabel('n')
plt.ylabel('Fraction of y')
plt.ylim(0,2)
plt.show()
print y




y1 = (sympy.Symbol('x1')**sympy.Symbol('n1'))/((sympy.Symbol('k1')**sympy.Symbol('n1'))+(sympy.Symbol('x1')**sympy.Symbol('n1')))
y1=y1.subs(sympy.Symbol('n1'), 5)
y1=y1.subs(sympy.Symbol('k1'), 10)
f1 = sympy.lambdify([sympy.Symbol('x1')], y1, 'numpy' )
x1_values = numpy.arange(0,100,1)
ss1 = x1_values.reshape(1,100)
print len(x1_values)
plt.subplot(211)
plt.plot(x1_values, f1(*ss1))
plt.xlabel('x')
plt.ylabel('Fraction of y')
plt.title('variable parameter: x')
plt.subplot(212)
plt.loglog(x1_values, f1(*ss1))
plt.xlabel('x')
plt.ylabel('Fraction of y')
plt.ylim(0,2)
plt.show()
plt.show()

print y1




y2 = (sympy.Symbol('x2')**sympy.Symbol('n2'))/((sympy.Symbol('k2')**sympy.Symbol('n2'))+(sympy.Symbol('x2')**sympy.Symbol('n2')))
y2=y2.subs(sympy.Symbol('x2'), 10)
y2=y2.subs(sympy.Symbol('n2'), 5)
f2 = sympy.lambdify([sympy.Symbol('k2')], y2, 'numpy' )
k2_values = numpy.arange(0,100,1)
ss2 = k2_values.reshape(1,100)
plt.subplot(211)
plt.plot(k2_values, f2(*ss2))
plt.xlabel('k')
plt.ylabel('Fraction of y')
plt.title('variable parameter: k')
plt.subplot(212)
plt.loglog(k2_values, f2(*ss2))
plt.xlabel('k')
plt.ylabel('Fraction of y')
plt.ylim(0,2)
plt.show()
print y2