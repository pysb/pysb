from pysb.integrate import odesolve
from pylab import *

from earm_1_0 import model


t = linspace(0, 6*3600, 1000)  # 6 hours, in seconds
y = odesolve(model, t)

tp = t / 3600  # x axis as hours

observations = ('Bid','PARP','mSmac')
y_norm = array([y[s] for s in observations]).T
y_norm = 1 - y_norm / y_norm.max(0)

#y_norm = array([y['tBid'], y['CPARP'], y['cSmac']]).T
#y_norm /= y_norm_2.max(0)

p = plot(tp, y_norm[:,0], 'b', tp, y_norm[:,1], 'y', tp, y_norm[:,2], 'r')
figlegend(p, observations, 'upper left')
a = gca()
a.set_ylim((-.05, 1.05))
show()
