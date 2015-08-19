from pysb.examples.simple_reaction_pyurdme import model
from pysb.tools.pysb_pyurdme import run_pyurdme
import numpy as np
import matplotlib.pyplot as plt
import pyurdme
from pysb.integrate import odesolve



model.diffusivities = [('E(b=None)',0.001), ('P(b=None)',0.2)]


initial_dist = {'E(b=None)':['set_initial_condition_place_near', [0.5,0.5]],  
                        'S(b=None)':['set_initial_condition_place_near', [1,0.5]]}

mesh = pyurdme.URDMEMesh.generate_unit_square_mesh(40,40)

tspan = np.linspace(0, 5, 500)
y=odesolve(model,tspan)
# plt.plot(tspan,y['__s2'])
# plt.show()
result = run_pyurdme(model, tspan, mesh, initial_dist)
