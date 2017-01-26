"""
Sensitivity analysis tools created to demonstrate GPU powered
analysis of PySB models.
Information can be found in Harris et. al, Bioinformatics.
Code written by James C. Pino

"""
import os
from itertools import product
import collections
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from pysb.simulator.scipyode import ScipyOdeSimulator
from pysb.bng import generate_equations
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
        # remove source species
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
        self.index_of_species_of_interest = self._create_index_of_species()
        self.simulations = self._setup_simulations()
        self.objective_function = objective_function
        self.observable = observable
        self.standard = None
        self.p_prime_matrix = np.zeros(self.size_of_matrix)
        self.p_matrix = np.zeros(self.size_of_matrix)

    def _calculate_objective(self, function_value):
        """ Calculates fraction of change for obj value and standard

        Parameters
        ----------
        function_value : scalar
            scalar value provided by objective function

        Returns
        -------

        """
        return (self.objective_function(function_value)
                - self.standard) / self.standard * 100.

    def run(self, run_solver, save_name=None, out_dir=None):
        """ Run function to perform sensitivity analysis

        Parameters
        ----------
        run_solver : function
            A solver function that returns the sensitivity analysis matrix
        save_name : str, optional
            prefix of
        out_dir : str, optional
            location to save output if required

        Returns
        -------

        """
        if out_dir is not None:
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

        solver = ScipyOdeSimulator(self.model, tspan=self.sim_time,
                                   integrator='lsoda',
                                   integrator_options={'rtol': 1e-8,
                                                       'atol': 1e-8,
                                                       'mxstep': 20000})
        sim_results = solver.run()
        self.standard = self.objective_function(
            np.array(sim_results.observables[self.observable]))
        sensitivity_output = run_solver(self.simulations)

        p_matrix = np.zeros(self.size_of_matrix)
        p_prime_matrix = np.zeros(self.size_of_matrix)
        counter = 0
        # places values in p matrix that are unique
        for i in range(len(p_matrix)):
            if i in self.b_index:
                tmp = self._calculate_objective(sensitivity_output[:, counter])
                p_matrix[i] = tmp
                counter += 1
        # places values in p matrix that are duplicated
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

        # save output if desired
        if save_name is not None:
            if out_dir is None:
                out_dir = '.'
            p_name = os.path.join(out_dir, '{}_p_matrix.csv'.format(save_name))
            p_prime_name = os.path.join(
                out_dir, '{}_p_prime_matrix.csv'.format(save_name))
            print("Saving p matrix and p' matrix to {} and {}".format(
                p_name, p_prime_name))

            np.savetxt(p_name, self.p_matrix)
            np.savetxt(p_prime_name, self.p_prime_matrix)

    def _create_index_of_species(self):
        """ create dictionary of initial conditions by index

        """
        index_of_init_condition = {}
        for i in range(len(self.model.initial_conditions)):
            for j in range(len(self.model.species)):
                if str(self.model.initial_conditions[i][0]) \
                        == str(self.model.species[j]):
                    x = self.model.initial_conditions[i][1].value
                    self.initial_conditions[j] = x
                    if self.model.initial_conditions[i][1].name \
                            in self.proteins_of_interest:
                        index_of_init_condition[
                            self.model.initial_conditions[i][1].name] = j
        index_of_species_of_interest = collections.OrderedDict(
            sorted(index_of_init_condition.items()))
        return index_of_species_of_interest

    def _g_function(self):
        """ creates sample matrix, index of samples values, and shows bij

        """
        counter = -1
        sampled_values_index = set()
        bij_unique = dict()
        sampling_matrix = np.zeros(
            (self.size_of_matrix, len(self.model.species)))
        sampling_matrix[:, :] = self.initial_conditions
        matrix = self.b_matrix
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

    def _setup_simulations(self):
        """ Creates initial conditions matrix for sensitivity analysis

        Returns
        -------

        """
        # create matrix (cartesian product of sample vals vs index of species
        a_matrix = cartesian_product(self.values_to_sample,
                                     self.index_of_species_of_interest)
        # reshape to flatten
        a_matrix = a_matrix.T.reshape(self.n_sam * self.n_proteins)
        # creates matrix b
        self.b_matrix = cartesian_product(a_matrix, a_matrix)

        # create matrix a'
        a_prime = cartesian_product(np.ones(self.n_sam),
                                    self.index_of_species_of_interest)
        a_prime = a_prime.T.reshape(self.n_sam * self.n_proteins)

        # creates matrix b prime
        self.b_prime_matrix = cartesian_product(a_prime, a_matrix)

        b_to_run, self.b_index, in_b = self._g_function()

        n_b_index = len(self.b_index)

        b_prime = np.zeros((self.size_of_matrix, len(self.model.species)))
        b_prime[:, :] = self.initial_conditions
        counter = -1

        bp_not_in_b_raw = set()
        bp_dict = dict()
        bp_not_in_b_dict = dict()
        bp_not_in_b_visited = dict()
        new_sim_counter = -1
        # checks for and removes duplicates of simulations initial conditions
        for j in range(len(self.b_prime_matrix)):
            for i in self.b_prime_matrix[j, :]:
                sigma_i, index_i = i[0]
                sigma_j, index_j = i[1]
                s_1 = str(i[0]) + str(i[1])
                counter += 1
                # no need for doing if same index
                if index_i == index_j:
                    continue
                # pointing to same simulation if already in b
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
        x = b_to_run[list(self.b_index)]
        y = b_prime[list(bp_not_in_b_raw)]
        simulations = np.vstack((x, y))
        print("Number of simulations to run = %s" % len(simulations))
        return simulations

    def create_plot_p_h_pprime(self, save_name=None, out_dir=None, show=True):
        """ creates a plot of P, H(B), and P'

        Parameters
        ----------
        save_name : str, optional
            name to save figure as
        out_dir : str, optional
            location to save figure
        show : bool
            show the plot

        Returns
        -------

        """
        colors = 'seismic'
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
        if save_name is not None:
            if out_dir is None:
                out_dir = '.'
            fig.savefig(os.path.join(out_dir,
                                     '{}_P_H_P_prime.png'.format(save_name)),
                        bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def create_individual_pairwise_plots(self, save_name=None, out_dir=None,
                                         show=True):
        """ creates a single plot containing heat plot of each specie pair

        Parameters
        ----------
        save_name : str, optional
            name ot save figure as
        out_dir : str, optional
            output directory
        show : bool
            show figure


        """
        colors = 'seismic'
        sens_ij_nm = []
        sens_matrix = self.p_matrix - self.p_prime_matrix
        v_max = max(np.abs(self.p_matrix.min()), self.p_matrix.max())
        v_min = -1 * v_max
        plt.figure(figsize=(self.n_proteins + 6, self.n_proteins + 6))
        gs = gridspec.GridSpec(self.n_proteins, self.n_proteins)
        # creates a plot of each species vs each species
        # adds space between plots so you can zoom in on output pairs
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

        if save_name is not None:
            if out_dir is None:
                out_dir = '.'
            plt.savefig(os.path.join(out_dir,
                                     '{}_subplots.png'.format(save_name)),
                        bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def create_boxplot_and_heatplot(self, x_axis_label=None, save_name=None,
                                    out_dir=None, show=True):
        """ Creates heat map and box plot of sensitivities


        Parameters
        ----------
        x_axis_label : str, optional
            label for x asis
        save_name : str, optional
            name of figure to save
        out_dir : str, option
            output directory to save figures
        show : bool
            Show plot

        Returns
        -------

        References
        -------
        Harris L. et al 2017

        """

        colors = 'seismic'
        sens_ij_nm = []
        sens_matrix = self.p_matrix - self.p_prime_matrix

        # separate each species sensitivity
        for j in range(0, self.nm, self.n_sam):
            per_protein1 = []
            for i in range(0, self.nm, self.n_sam):
                if i != j:
                    tmp = sens_matrix[j:j + self.n_sam,
                                      i:i + self.n_sam].copy()
                    per_protein1.append(tmp)
            sens_ij_nm.append(per_protein1)

        # Create heatmap and boxplot of data
        plt.figure(figsize=(14, 10))
        plt.subplots_adjust(hspace=0.1)

        # use gridspec to scale colorbar nicely
        outer = gridspec.GridSpec(2, 1, width_ratios=[0.4, 1.],
                                  height_ratios=[0.03, 1])

        gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1],
                                               hspace=.35)
        ax0 = plt.subplot(gs1[0])
        ax1 = plt.subplot(gs2[0])

        # scale the colors to minimum or maximum of p matrix
        v_max = max(np.abs(self.p_matrix.min()), self.p_matrix.max())
        v_min = -1 * v_max

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
        if x_axis_label is not None:
            ax2.set_xlabel(x_axis_label, fontsize=12)
        plt.setp(ax2, yticklabels=reversed(self.proteins_of_interest))
        ax2.yaxis.tick_left()
        ax2.set_aspect(1. / ax2.get_data_ratio(), adjustable='box', )
        if save_name is not None:
            if out_dir is None:
                out_dir = '.'
            plt.savefig(os.path.join(out_dir, save_name + '.png'),
                        bbox_inches='tight')
            plt.savefig(os.path.join(out_dir, save_name + '.eps'),
                        bbox_inches='tight')
            plt.savefig(os.path.join(out_dir, save_name + '.svg'),
                        bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


def cartesian_product(array_1, array_2):
    """ creates cartesian product between two lists

    Parameters
    ----------
    array_1 : list_like
    array_2 : list_like

    Returns
    -------
    out : np.array
        array of shape (len(array_1) x len(array_2))
    """
    a = list(product(array_1, array_2))
    a = np.asarray(a, dtype=','.join('object' for _ in range(len(a[0]))))
    return a.reshape(len(array_1), len(array_2))
