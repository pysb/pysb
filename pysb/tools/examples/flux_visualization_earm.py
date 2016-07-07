from __future__ import print_function
from earm.lopez_embedded import model
from pysb.tools.model_visualization import run_visualization
import numpy as np
import csv

# type Bax cluster: 5400
# type Bak cluster: 4052

f = open('/home/oscar/Documents/tropical_project/parameters_5000/pars_embedded_4052.txt')
data = csv.reader(f)
parames = [float(i[1]) for i in data]

tspan = np.linspace(0, 20160, 100)

run_visualization(model, tspan, parameters=parames, render_type='flux', verbose=False)
print('finished')
