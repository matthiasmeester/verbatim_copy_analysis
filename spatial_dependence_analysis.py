from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from src.verbatim_heat_map_creator import VerbatimHeatMapCreator

# File for generating plots, very messy.
max_range = int(sqrt(200 ** 2 + 200 * 2))
threshold_range = 3
inv_dist_weight_exp = 2
sim_type = 'stone'
directory = "simulations"
k = '1.0'
n = 5
filter_radius = 100

file = np.load(f'simulations/qsSim_{sim_type}_{k}.npz')
ti = file['ti']

index_map = file['indexMap'][n - 1, :, :]

# # --- Do spatial dependency analysis ---
heat_map_creator = VerbatimHeatMapCreator(index_map)
neighbourhood_verbatim, distances = heat_map_creator.spatial_dependency_analysis(filter_radius=filter_radius)

# --- Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle(f'SDA - {sim_type}, k={k}, n={n}', size='xx-large')
ax1.set_title('Probability verbatim at distance histogram')
ax1.set_xlabel('Distance')
ax1.set_ylabel('Probability')
ax1.bar(list(distances.keys()), list(distances.values()))
neigh_img = ax2.imshow(neighbourhood_verbatim, extent=[-filter_radius, filter_radius, -filter_radius, filter_radius])
fig.colorbar(neigh_img, ax=ax2)
ax2.set_title('Mean kernel verbatim pattern')
ax2.set_xlabel('X distance')
ax2.set_ylabel('Y distance')
plt.savefig(f'output/spatial_dependency/k={k}, n={n}.png', bbox_inches='tight', dpi=150)
plt.show()
