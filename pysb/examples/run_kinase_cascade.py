from __future__ import print_function
from pysb.simulator import ScipyOdeSimulator
from matplotlib.pyplot import plot, legend, show
from matplotlib.pylab import linspace
from kinase_cascade import model


tspan = linspace(0, 1200)
print("Simulating...")
yfull = ScipyOdeSimulator(model).run(tspan=tspan).all
plot(tspan, yfull['ppMEK'], label='ppMEK')
plot(tspan, yfull['ppERK'], label='ppERK')
legend(loc='upper left')
show()
