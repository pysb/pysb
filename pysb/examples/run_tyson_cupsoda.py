from pysb.examples.tyson_oscillator import model
from pysb.simulators.cupsoda import *
import numpy as np
import matplotlib.pyplot as plt

tspan = np.linspace(0, 500, 501)

set_cupsoda_path("/Users/lopezlab/cupSODA") #FIXME: should search for cupSODA in standard locations
solver = CupSodaSolver(model, tspan, atol=1e-12, rtol=1e-6, max_steps=20000, verbose=True)

n_sims = 100

# Rate constants
param_values = np.ones((n_sims, len(model.parameters_rules())))
for i in range(len(param_values)):
    for j in range(len(param_values[i])):
        param_values[i][j] *= model.parameters_rules()[j].value

# Initial concentrations
y0 = np.zeros((n_sims, len(model.species)))
for i in range(len(y0)):
    for ic in model.initial_conditions:
        for j in range(len(y0[i])):
            if str(ic[0]) == str(model.species[j]):
                y0[i][j] = ic[1].value
                break

solver.run(param_values, y0) #, outdir=os.path.join(outdir,'NSAMPLES_'+str(n_samples))) #obs_species_only=False, load_conc_data=False)
        
