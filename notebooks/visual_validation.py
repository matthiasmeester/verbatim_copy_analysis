import math
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from src.heat_map_analysis import HeatMapAnalysis
from src.verbatim_heat_map_creator import VerbatimHeatMapCreator

# File for generating plots, very messy.

max_range = int(sqrt(200 ** 2 + 200 * 2))
threshold_range = 3
inv_dist_weight_exp = 2
sim_type = 'stone'
directory = "simulations"
k = '1.0'
n = 199
filter_radius = 1
max_noise_heat_l = 0.00010004966211932229
max_noise_heat_s = 0.00246278367601897

file = np.load(f'simulations/qsSim_{sim_type}_{k}.npz')
ti = file['ti']

index_map = file['indexMap'][n - 1, :, :]
simulation = file['sim'][n - 1, :, :]
sourceIndex = np.stack(np.meshgrid(np.arange(ti.shape[0]) / ti.shape[0], np.arange(ti.shape[1]) / ti.shape[1]) + [np.ones_like(ti)], axis=-1)

simulation_size = index_map.shape[0] * index_map.shape[1]


def _index_to_coord(index):
    index = int(index)
    y = math.floor(index / index_map.shape[0])
    x = index - y * index_map.shape[1]
    return int(x), int(y)


def simulation_indices_to_training_indices(simulation_indices):
    result = np.zeros((200, 200))
    for iy, ix in np.ndindex(simulation_indices.shape):
        simulation_index = simulation_indices[iy, ix]
        x, y = _index_to_coord(simulation_index)
        result[y, x] = simulation_index

    return result


# # --- Do filter analysis ---
heat_map_creator = VerbatimHeatMapCreator(index_map)
heat_map = heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, inv_dist_weight_exp)
non_weighted_heat_map_s = heat_map_creator.get_verbatim_heat_map_filter_basis(threshold_range, 0)
non_weighted_heat_map_l = heat_map_creator.get_verbatim_heat_map_filter_basis(max_range, 0)
#
# # --- Calculate statistics ---
# dist_weighted_mean_heat_value = HeatMapAnalysis(heat_map).mean_heat_value()
proportion_above_stat_s = HeatMapAnalysis(non_weighted_heat_map_s).above_treshold_heat_map(100 * max_noise_heat_s)
proportion_above_stat_l = HeatMapAnalysis(non_weighted_heat_map_l).above_treshold_heat_map(100 * max_noise_heat_l)
# Critical area:
critical_verbatim_area = (proportion_above_stat_l == 1) & (proportion_above_stat_s == 1)

filtered_sim = np.where(critical_verbatim_area, simulation, 0)
ti_indices = simulation_indices_to_training_indices(np.where(critical_verbatim_area, index_map, 0))

#  --- Do plotting ---
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 9))
ax1.imshow(ti, interpolation='none')
ax1.set_title('Training image')
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
ax2.imshow(simulation, interpolation='none')
ax2.set_title(f'QSsim \'{sim_type}\' $k={k}$, $n={n}$')
ax3.set_title('Source verbatim patches')
ax3.imshow(np.where(ti_indices, ti, 0), interpolation='none')

ax4.imshow(filtered_sim, interpolation='none')
ax4.set_title('Simulation verbatim patches')

fig.suptitle('Visual validation of verbatim patches, $TNA_s = 1, TNA_l = 1$')
plt.savefig(f'output/visual_validation/k={k}, n={n}.png', bbox_inches='tight', dpi=150)
plt.show()
