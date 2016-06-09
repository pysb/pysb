import os
import time
import numpy as np
import scipy.interpolate
import pysb
from pysb.bng import generate_equations
from pysb.integrate import odesolve
from pysb.util import update_param_vals, load_params
#from pysb_cupsoda import set_cupsoda_path, CupsodaSolver



#set_cupsoda_path("/home/pinojc/git/cupSODA")
run = 'cupSODA'
run = 'scipy'

ATOL = 1e-8
RTOL = 1e-8
mxstep = 20000
det = 1
vol = 10e-20


class SensitivityAnalysis:
    def __init__(self, model, tspan, values_to_sample, objective_function, observable):
        self.model = model
        self.tspan = tspan
        generate_equations(self.model)
        self.proteins_of_interest = list((i[1].name for i in model.initial_conditions))
        if '__source_0' in self.proteins_of_interest:
            self.proteins_of_interest.remove('__source_0')
        print self.proteins_of_interest
        self.n_proteins = len(self.proteins_of_interest)
        self.nominal_values = np.array([p.value for p in model.parameters])
        self.vals = values_to_sample
        self.n_sam = len(self.vals)
        self.size_of_matrix = (self.n_proteins * self.n_proteins - self.n_proteins) * (self.n_sam * self.n_sam) / 2
        # Create parameter matrix from model parameters
        self.c_matrix = np.zeros((self.size_of_matrix, len(self.nominal_values)))
        self.c_matrix[:, :] = self.nominal_values
        self.MX_0 = np.zeros((self.size_of_matrix, len(model.species)))
        self.objective_function = objective_function
        self.observable = observable
        self.standard = None

    def create_initial_concentration_matrix(self):
        index_of_species_of_interest = {}
        for i in range(len(self.model.initial_conditions)):
            for j in range(len(self.model.species)):
                if str(self.model.initial_conditions[i][0]) == str(self.model.species[j]):
                    x = self.model.initial_conditions[i][1].value
                    self.MX_0[:, j] = x
                    if self.model.initial_conditions[i][1].name in self.proteins_of_interest:
                        index_of_species_of_interest[self.model.initial_conditions[i][1].name] = j
        counter = 0
        done = []
        for i in self.proteins_of_interest:
            for j in self.proteins_of_interest:
                if j in done:
                    continue
                if i == j:
                    continue
                for a, c in enumerate(self.vals):
                    for b, d in enumerate(self.vals):
                        x = index_of_species_of_interest[i]
                        y = index_of_species_of_interest[j]
                        self.MX_0[counter, x] *= c
                        self.MX_0[counter, y] *= d
                        counter += 1
            done.append(i)
        print("Number of simulations to run = %s" % counter)

    def calculate_objective(self,species):
        return (self.objective_function(species)-self.standard)/self.standard*100

    def run(self, option, save_name, output_directory):
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        self.create_initial_concentration_matrix()
        solver = pysb.integrate.Solver(self.model, self.tspan, rtol=RTOL, atol=ATOL, integrator='lsoda', mxstep=mxstep)
        solver.run()
        self.standard = self.objective_function(solver.yobs[self.observable])

        if option == 'scipy':
            solver = pysb.integrate.Solver(self.model, self.tspan, rtol=RTOL, atol=ATOL,
                                           integrator='lsoda', mxstep=mxstep)
            sensitivity_matrix = np.zeros((len(self.tspan), self.size_of_matrix))
            start_time = time.time()
            for k in range(self.size_of_matrix):
                solver.run(y0=self.MX_0[k, :])
                sensitivity_matrix[:,k] = solver.yobs[self.observable]
                #np.savetxt(os.path.join(output_directory, str(k)), solver.yobs[self.observable])
            time_taken = time.time() - start_time
            print('sim = %s , time = %s sec' % (self.size_of_matrix, time_taken))
            image = np.zeros((self.n_proteins * self.n_sam, self.n_proteins * self.n_sam))
            counter = 0
            for i in range(len(self.proteins_of_interest)):
                y = i * len(self.vals)
                for j in range(i, len(self.proteins_of_interest)):
                    x = j * self.n_sam
                    if x == y:
                        continue
                    for a in range(self.n_sam):
                        for b in range(self.n_sam):
                            tmp = self.calculate_objective(sensitivity_matrix[:, counter])
                            image[y + a, x + b] = tmp
                            image[x + b, y + a] = tmp
                            counter += 1
            np.savetxt(os.path.join(output_directory,'%s_image_matrix.csv' % save_name), image)
        # if option == "cupSODA":
        #     solver = CupsodaSolver(self.model, self.tspan, atol=ATOL, rtol=RTOL, verbose=False)
        #     start_time = time.time()
        #     solver.run(self.c_matrix,
        #                self.MX_0,
        #                gpu=0,
        #                max_steps=mxstep,
        #                obs_species_only=True,
        #                memory_usage='shared',
        #                vol=vol)
        #     time_taken = time.time() - start_time
        #     print('sim = %s , time = %s sec' % (self.size_of_matrix, time_taken))
        #     print('out==', solver.yobs[0][0], solver.yobs[0][-1], '==out')
        #     sensitivity_matrix = np.zeros((len(self.tspan), self.size_of_matrix))
        #     counter = 0
        #     for i in range(self.n_proteins):
        #         for j in range(i, self.n_proteins):
        #             if i == j:
        #                 continue
        #             for a in range(self.n_sam):
        #                 for b in range(self.n_sam):
        #                     sensitivity_matrix[:, counter] = solver.yobs[counter][self.observable]
        #                     counter += 1
        #     image = np.zeros((self.n_proteins * self.n_sam, self.n_proteins * self.n_sam))
        #     counter = 0
        #     for i in range(len(self.proteins_of_interest)):
        #         y = i * len(self.vals)
        #         for j in range(i, len(self.proteins_of_interest)):
        #             x = j * self.n_sam
        #             if x == y:
        #                 continue
        #             for a in range(self.n_sam):
        #                 for b in range(self.n_sam):
        #                     tmp = self.calculate_objective(sensitivity_matrix[:, counter])
        #                     image[y + a, x + b] = tmp
        #                     image[x + b, y + a] = tmp
        #                     counter += 1
        #     np.savetxt('%s_image_matrix.csv' % save_name, image)
        #     print('Saving image')