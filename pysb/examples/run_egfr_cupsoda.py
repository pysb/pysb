try:
    import chen_sorger_2009_egfr_simplified
except ImportError:
    raise ImportError('Please install EGFR model from '
                      'https://github.com/LoLab-VU/egfr and add it to your '
                      'PYTHONPATH.')
from chen_sorger_2009_egfr_simplified.erbb_exec import model
from pysb.simulators.cupsoda import CupSodaSolver
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

observables = ['obsAKTPP', 'obsErbB1_P_CE', 'obsERKPP']
colors = ['b', 'g', 'r']

model_dir = os.path.dirname(chen_sorger_2009_egfr_simplified.__file__)

expdata = np.load(os.path.join(model_dir,
                 'experimental_data/experimental_data_A431_highEGF_unnorm'
                 '.npy'))
exptimepts = [0, 149, 299, 449, 599, 899, 1799, 2699, 3599, 7199]

tspan = np.linspace(0, 8000, 8001)

# Sub best-fit parameters
with open(os.path.join(model_dir,
        'calibration_files'
        '/A431_highEGF_erbb_simplified_fitted_values_no_ERKPP_fit.p'),
                       'rb') as handle:
     fittedparams = pickle.loads(handle.read())
for p in model.parameters:
    if p.name in fittedparams:
        p.value = fittedparams[p.name]

#####
# from pysb.integrate import odesolve
# x = odesolve(model, tspan, verbose=True)
# for i in range(len(observables)):
#     plt.plot(tspan, x[observables[i]], lw=3, label=observables[i],
#              color=colors[i])
# #     plt.plot(exptimepts, expdata[:,i], '*', lw=0, ms=12, mew=0,
#                mfc=colors[i])
# plt.legend(loc=0)
# plt.ylim(ymin=1)
# plt.ylabel('molecules')
# plt.xlabel('time')
# plt.show()
# quit()
#####

solver = CupSodaSolver(model, tspan, atol=1e-12, rtol=1e-6, max_steps=20000,
                       verbose=True)

n_sims = 1

# Rate constants
param_values = np.ones((n_sims, len(model.parameters)))
for i in range(len(param_values)):
    for j in range(len(param_values[i])):
        param_values[i][j] *= model.parameters[j].value

# Initial concentrations
y0 = np.zeros((n_sims, len(model.species)))
for i in range(len(y0)):
    for ic in model.initial_conditions:
        for j in range(len(y0[i])):
            if str(ic[0]) == str(model.species[j]):
                y0[i][j] = ic[1].value
                break

solver.run(param_values=param_values, y0=y0)
yfull = solver.get_yfull()

for i in range(len(observables)):
    plt.plot(tspan, yfull[observables[i]][0], lw=3, label=observables[i],
             color=colors[i])
    plt.plot(exptimepts, expdata[:,i], '*', lw=0, ms=12, mew=0, mfc=colors[i])
plt.legend(loc=0)
plt.ylim(ymin=1)
plt.ylabel('molecules')
plt.xlabel('time')
plt.show()
