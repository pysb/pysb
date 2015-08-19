from pysb.examples.tyson_oscillator import model 
from pysb.tools.max_monomials import run_tropical
import numpy as np
import matplotlib.pyplot as plt


tspan = np.linspace(0, 100, 100)
run_tropical(model, tspan, sp_visualize=[3,5], stoch=True)
run_tropical(model, tspan, sp_visualize=[3,5], stoch=False)

