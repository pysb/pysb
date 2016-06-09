from pysb.tools.sensitivity_analysis import SensitivityAnalysis
from pysb.examples.tyson_oscillator import model
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid
import os

tspan = np.linspace(0, 200, 5001)

def obj_func_cell_cycle(out):
    timestep = tspan[:-1]
    y = out[:-1] - out[1:]
    freq = 0
    local_times = []
    prev = y[0]
    for n in range(1, len(y)):
        if y[n] > 0 > prev:
            local_times.append(timestep[n])
            freq += 1
        prev = y[n]

    local_times = np.array(local_times)
    local_freq = np.average(local_times)/len(local_times)*2
    return local_freq


def create_boxplot_and_heatplot(model,values, data, x_axis_label, savename):
    proteins_of_interest = []
    for i in model.initial_conditions:
        proteins_of_interest.append(i[1].name)

    colors = 'seismic'

    sens_matrix = np.loadtxt(data)
    length_values = len(values)
    length_image = len(sens_matrix)
    median = int(np.median(range(0, length_values)))
    sens_ij_nm = []
    for j in range(0, length_image, length_values):
        per_protein1 = []
        for i in range(0, length_image, length_values):
            if i == j:
                continue
            tmp = sens_matrix[j:j + length_values, i:i + length_values].copy()
            tmp -= tmp[median, :]  # sens_ij_0m
            per_protein1.append(tmp)
        sens_ij_nm.append(per_protein1)

    v_max = max(np.abs(sens_matrix.min()), sens_matrix.max())
    v_min = -1 * v_max

    fig = plt.figure(figsize=(5, 10))
    ax1 = fig.add_subplot(2, 1, 1)

    im = ax1.imshow(sens_matrix, interpolation='nearest', origin='lower', cmap=plt.get_cmap(colors), vmin=v_min,
                    vmax=v_max, extent=[0, length_image, 0, length_image])
    shape_label = np.arange(length_values / 2, length_image, length_values)
    plt.xticks(shape_label, proteins_of_interest, rotation='vertical', fontsize=12)
    plt.yticks(shape_label, proteins_of_interest, fontsize=12)
    xticks = ([i for i in range(0, length_image, length_values)])
    ax1.set_xticks(xticks, minor=True)
    ax1.set_yticks(xticks, minor=True)
    plt.grid(True, which='minor', axis='both', linestyle='--')
    divider = axgrid.make_axes_locatable(ax1)
    cax = divider.append_axes("top", size="5%", pad=0.3)
    cax.tick_params(labelsize=12)

    if savename == 'tyson_sensitivity_boxplot.png':
        ticks = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

    color_bar = fig.colorbar(im, cax=cax, ticks=ticks, orientation='horizontal')
    color_bar.set_ticks(ticks)
    color_bar.ax.set_xticklabels(ticks)
    color_bar.set_label('% change', labelpad=-40, y=0.45)
    ax2 = plt.subplot(2, 1, 2)
    ax2.boxplot(sens_ij_nm, vert=False, labels=None, showfliers=False)
    ax2.set_xlabel(x_axis_label, fontsize=12)
    plt.setp(ax2, yticklabels=proteins_of_interest)
    ax2.yaxis.tick_left()
    ax1.annotate('A', xy=(0, 1), xycoords='axes fraction', fontsize=16,
                 xytext=(-55, 75), textcoords='offset points',
                 ha='left', va='top')
    ax2.annotate('B', xy=(0, 1), xycoords='axes fraction', fontsize=16,
                 xytext=(-25, 25), textcoords='offset points',
                 ha='left', va='top')
    plt.tight_layout(h_pad=2.5)
    plt.subplots_adjust(top=0.9)
    plt.savefig(savename, bbox_tight='True')
    plt.show()


def test_sens():

    observable = 'Y3'
    savename = 'here_here'
    vals = np.linspace(.8, 1.2, 21)
    sens = SensitivityAnalysis(model, tspan, vals, obj_func_cell_cycle, observable,)
    output_dir = 'test'
    sens.run(option='scipy',save_name=savename,output_directory=output_dir)
    create_boxplot_and_heatplot(model,vals, os.path.join(output_dir, '%s_image_matrix.csv' %(savename)),
                                'Percent change in period','tyson_sensitivity_boxplot.png')
test_sens()




