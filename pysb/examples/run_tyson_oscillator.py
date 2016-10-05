from pysb.examples.tyson_oscillator import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator
import matplotlib.pyplot as plt

t = np.linspace(0, 100, 10001)
x = ScipyOdeSimulator(model).run(tspan=t).all

plt.plot(t, x['CT'],  lw=2, label='CT')  # Good validation of mass balance for cdc2, should be constant at 1
plt.plot(t, x['YT'], lw=2, label='YT')
plt.plot(t, x['M'], lw=2, label='M')

plt.legend(loc=0)
plt.xlabel('time')
plt.ylabel('population')

plt.show()
