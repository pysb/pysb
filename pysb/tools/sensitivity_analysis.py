"""
Sensitivity analysis tools created to demonstrate GPU powered
analysis of PySB models.
Information can be found in Harris et. al, Bioinformatics.
Code written by James C. Pino

"""
import os
from itertools import product
import collections
import matplotlib
from matplotlib import gridspec, pyplot
import numpy as np
import pysb.integrate
from pysb.bng import generate_equations
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.use('Agg')
plt = matplotlib.pyplot


class InitialConcentrationSensitivityAnalysis:
    """ Performs pairwise sensitivity analysis of initial conditions

    """

    def __init__(self, model, time_span, values_to_sample, objective_function,
                 observable):
        self.model = model
        self.sim_time = time_span
        generate_equations(self.model)
        self.proteins_of_interest = list(
                (i[1].name for i in model.initial_conditions))
        if '__source_0' in self.proteins_of_interest:
            self.proteins_of_interest.remove('__source_0')
        self.proteins_of_interest = sorted(self.proteins_of_interest)
        self.n_proteins = len(self.proteins_of_interest)
        self.values_to_sample = values_to_sample
        self.n_sam = len(self.values_to_sample)
        self.b_matrix = []
        self.b_prime_matrix = []
        self.nm = self.n_proteins * self.n_sam
        self.size_of_matrix = self.nm ** 2
        self.shape_of_matrix = (self.nm, self.nm)
        self.initial_conditions = np.zeros(len(self.model.species))
        self.index_of_species_of_interest = self.create_index_of_species()
        self.setup_sampling_matrix()
        self.simulations = self._find_redundant_simulations()
        self.objective_function = objective_function
        self.observable = observable
        self.standard = None
        self.p_prime_matrix = np.zeros(self.size_of_matrix)
        self.p_matrix = np.zeros(self.size_of_matrix)

    def _calculate_objective(self, species):
        """ Calculates fraction of change between standard value and newly obtained value

        :param species:
        :return: fraction of change, type float
        """
        return (self.objective_function(
                species) - self.standard) / self.standard * 100.

    def run(self, run_solver, save_name, out_dir):
        """

        :param run_solver:
        :param save_name:
        :param out_dir:
        """
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        solver = pysb.integrate.Solver(self.model, self.sim_time, rtol=1e-8,
                                       atol=1e-8, integrator='lsoda',
                                       mxstep=20000)
        solver.run()
        self.standard = self.objective_function(solver.yobs[self.observable])
        sensitivity_output = run_solver(self.simulations)

        p_matrix = np.zeros(self.size_of_matrix)
        p_prime_matrix = np.zeros(self.size_of_matrix)
        counter = 0
        for i in range(len(p_matrix)):
            if i in self.b_index:
                tmp = self._calculate_objective(sensitivity_output[:, counter])
                p_matrix[i] = tmp
                counter += 1

        for i in range(len(p_matrix)):
            if i in self.b_prime_not_in_b:
                tmp = self._calculate_objective(
                        sensitivity_output[:, self.b_prime_not_in_b[i]])
                p_prime_matrix[i] = tmp
            elif i in self.b_prime_in_b:
                p_prime_matrix[i] = p_matrix[self.b_prime_in_b[i]]
        # Reshape
        p_matrix = p_matrix.reshape(self.shape_of_matrix)
        # Project the mirrored image
        self.p_matrix = p_matrix + p_matrix.T
        self.p_prime_matrix = p_prime_matrix.reshape(self.shape_of_matrix)

        np.savetxt(os.path.join(out_dir, '%s_p_matrix.csv' % save_name),
                   self.p_matrix)
        np.savetxt(os.path.join(out_dir, '%s_p_prime_matrix.csv' % save_name),
                   self.p_prime_matrix)

    def create_index_of_species(self):
        """

        :return:
        """
        index_of_init_condition = {}
        for i in range(len(self.model.initial_conditions)):
            for j in range(len(self.model.species)):
                if str(self.model.initial_conditions[i][0]) == str(
                        self.model.species[j]):
                    x = self.model.initial_conditions[i][1].value
                    self.initial_conditions[j] = x
                    if self.model.initial_conditions[i][
                        1].name in self.proteins_of_interest:
                        index_of_init_condition[
                            self.model.initial_conditions[i][1].name] = j
        index_of_species_of_interest = collections.OrderedDict(
                sorted(index_of_init_condition.items()))
        return index_of_species_of_interest

    def setup_sampling_matrix(self):
        """

        :return:
        """

        a_matrix = cartesian_product(self.values_to_sample,
                                     self.index_of_species_of_interest)
        a_matrix = a_matrix.T.reshape(self.n_sam * self.n_proteins)

        a_prime = cartesian_product(np.ones(self.n_sam),
                                    self.index_of_species_of_interest)
        a_prime = a_prime.T.reshape(self.n_sam * self.n_proteins)

        self.b_matrix = cartesian_product(a_matrix, a_matrix)
        self.b_prime_matrix = cartesian_product(a_prime, a_matrix)

    def _g_function(self, matrix):
        """

        :param matrix:
        :return:
        """
        counter = -1
        sampled_values_index = set()
        bij_unique = dict()
        sampling_matrix = np.zeros(
                (self.size_of_matrix, len(self.model.species)))
        sampling_matrix[:, :] = self.initial_conditions
        for j in range(len(matrix)):
            for i in matrix[j, :]:
                sigma_i, index_i = i[0]
                sigma_j, index_j = i[1]
                s_1 = str(i[0]) + str(i[1])
                s_2 = str(i[1]) + str(i[0])
                counter += 1
                if index_i == index_j:
                    continue
                elif s_1 in bij_unique or s_2 in bij_unique:
                    continue
                else:
                    x = self.index_of_species_of_interest[index_i]
                    y = self.index_of_species_of_interest[index_j]
                    sampling_matrix[counter, x] *= sigma_i
                    sampling_matrix[counter, y] *= sigma_j
                    bij_unique[s_1] = counter
                    bij_unique[s_2] = counter
                    sampled_values_index.add(counter)

        return sampling_matrix, sampled_values_index, bij_unique

    def _find_redundant_simulations(self):
        """ Finds and removes redundant simulations

        :return: simulations to run, np.array of initial conditions
        """
        self.b_to_run, self.b_index, in_b = self._g_function(self.b_matrix)

        n_b_index = len(self.b_index)

        b_prime = np.zeros((self.size_of_matrix, len(self.model.species)))
        b_prime[:, :] = self.initial_conditions
        counter = -1

        bp_not_in_b_raw = set()
        bp_dict = dict()
        bp_not_in_b_dict = dict()
        bp_not_in_b_visited = dict()
        new_sim_counter = -1
        for j in range(len(self.b_prime_matrix)):
            for i in self.b_prime_matrix[j, :]:
                sigma_i, index_i = i[0]
                sigma_j, index_j = i[1]
                s_1 = str(i[0]) + str(i[1])
                counter += 1
                if index_i == index_j:
                    continue
                elif s_1 in in_b:
                    bp_dict[counter] = in_b[s_1]
                elif s_1 in bp_not_in_b_visited:
                    bp_not_in_b_dict[counter] = bp_not_in_b_visited[s_1]
                else:
                    new_sim_counter += 1
                    x = self.index_of_species_of_interest[index_i]
                    y = self.index_of_species_of_interest[index_j]
                    b_prime[new_sim_counter, x] *= sigma_i
                    b_prime[new_sim_counter, y] *= sigma_j
                    bp_not_in_b_visited[s_1] = new_sim_counter + n_b_index
                    bp_not_in_b_dict[counter] = new_sim_counter + n_b_index
                    bp_not_in_b_raw.add(new_sim_counter)

        self.b_prime_in_b = bp_dict
        self.b_prime_not_in_b = bp_not_in_b_dict
        x = self.b_to_run[list(self.b_index)]
        y = b_prime[list(bp_not_in_b_raw)]
        simulations = np.vstack((x, y))
        print("Number of simulations to run = %s" % len(simulations))
        return simulations

    def create_boxplot_and_heatplot(self, x_axis_label='', savename='tmp',
                                    out_dir='.'):
        """

        :param x_axis_label:
        :param savename:
        :param out_dir:
        """
        colors = 'seismic'
        sens_ij_nm = []
        sens_matrix = self.p_matrix - self.p_prime_matrix
        v_max = max(np.abs(self.p_matrix.min()), self.p_matrix.max())
        v_min = -1 * v_max

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.imshow(self.p_matrix, interpolation='nearest', origin='upper',
                   cmap=plt.get_cmap(colors), vmin=v_min, vmax=v_max,
                   extent=[0, self.nm, 0, self.nm])

        ax2.imshow(self.p_prime_matrix, interpolation='nearest',
                   origin='upper', cmap=plt.get_cmap(colors),
                   vmin=v_min, vmax=v_max, extent=[0, self.nm, 0, self.nm])

        ax3.imshow(sens_matrix, interpolation='nearest', origin='upper',
                   cmap=plt.get_cmap(colors), vmin=v_min, vmax=v_max)

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax1.set_title('P', fontsize=22)
        ax2.set_title('H(B\')', fontsize=22)
        ax3.set_title('P\' = P - H(B\')', fontsize=22)
        fig.subplots_adjust(wspace=0, hspace=0.0)
        fig.savefig(os.path.join(out_dir,
                                 '{}_side_by_side.png'.format(savename)),
                    bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(self.n_proteins + 6, self.n_proteins + 6))
        gs = gridspec.GridSpec(self.n_proteins, self.n_proteins)

        for n, j in enumerate(range(0, self.nm, self.n_sam)):
            per_protein1 = []
            for m, i in enumerate(range(0, self.nm, self.n_sam)):
                ax2 = plt.subplot(gs[n, m])
                if n == 0:
                    ax2.set_xlabel(self.proteins_of_interest[m], fontsize=20)
                    ax2.xaxis.set_label_position('top')
                if m == 0:
                    ax2.set_ylabel(self.proteins_of_interest[n], fontsize=20)
                plt.xticks([])
                plt.yticks([])
                if i != j:
                    tmp = sens_matrix[j:j + self.n_sam,
                          i:i + self.n_sam].copy()
                    ax2.imshow(tmp, interpolation='nearest', origin='upper',
                               cmap=plt.get_cmap(colors), vmin=v_min,
                               vmax=v_max)
                    per_protein1.append(tmp)
                else:
                    ax2.imshow(np.zeros((self.n_sam, self.n_sam)),
                               interpolation='nearest', origin='upper',
                               cmap=plt.get_cmap(colors), vmin=v_min,
                               vmax=v_max)
            sens_ij_nm.append(per_protein1)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '{}_subplots.png'.format(savename)),
                    bbox_inches='tight')
        plt.close()

        # Create heatmap and boxplot of data
        plt.figure(figsize=(14, 10))
        plt.subplots_adjust(hspace=0.1)
        outer = gridspec.GridSpec(2, 1, width_ratios=[0.4, 1],
                                  height_ratios=[0.03, 1])

        gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1],
                                               hspace=.35)
        ax0 = plt.subplot(gs1[0])
        ax1 = plt.subplot(gs2[0])
        # create heatmap of sensitivities
        im = ax1.imshow(self.p_matrix, interpolation='nearest', origin='upper',
                        cmap=plt.get_cmap(colors), vmin=v_min,
                        vmax=v_max, extent=[0, self.nm, 0, self.nm])

        shape_label = np.arange(self.n_sam / 2, self.nm, self.n_sam)
        plt.xticks(shape_label, self.proteins_of_interest, rotation='vertical',
                   fontsize=12)
        plt.yticks(shape_label, reversed(self.proteins_of_interest),
                   fontsize=12)
        x_ticks = ([i for i in range(0, self.nm, self.n_sam)])
        ax1.set_xticks(x_ticks, minor=True)
        ax1.set_yticks(x_ticks, minor=True)
        plt.grid(True, which='minor', linestyle='--')
        color_bar = plt.colorbar(im, cax=ax0, orientation='horizontal',
                                 use_gridspec=True)
        color_bar.set_label('% change', y=1, labelpad=5)
        color_bar.ax.xaxis.set_label_position('top')
        ticks = np.linspace(v_min, v_max, 5, dtype=int)
        color_bar.set_ticks(ticks)
        color_bar.ax.set_xticklabels(ticks)

        # create boxplot of single parameter sensitivities
        ax2 = plt.subplot(gs2[1])
        ax2.boxplot(sens_ij_nm[::-1], vert=False, labels=None, showfliers=True,
                    whis='range')
        ax2.set_xlim(v_min - 2, v_max + 2)
        ax2.set_xlabel(x_axis_label, fontsize=12)
        plt.setp(ax2, yticklabels=reversed(self.proteins_of_interest))
        ax2.yaxis.tick_left()
        ax2.set_aspect(1. / ax2.get_data_ratio(), adjustable='box', )
        plt.savefig(os.path.join(out_dir, savename + '.png'),
                    bbox_inches='tight')
        plt.savefig(os.path.join(out_dir, savename + '.eps'),
                    bbox_inches='tight')
        plt.savefig(os.path.join(out_dir, savename + '.svg'),
                    bbox_inches='tight')
        plt.show()


def cartesian_product(array_1, array_2):
    """

    :param array_1:
    :param array_2:
    :return:
    """
    a = list(product(array_1, array_2))
    a = np.asarray(a, dtype=','.join('object' for _ in range(len(a[0]))))
    return a.reshape(len(array_1), len(array_2))
