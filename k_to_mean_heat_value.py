import os.path
from random import seed

import matplotlib.pyplot as plt
import numpy as np

from src.heat_map_analysis import HeatMapAnalysis
from src.verbatim_heat_map_creator import VerbatimHeatMapCreator

# ----- Verbatim copy statistic: -----
# --- Custom variables ---
inv_dist_weight_exp = 2
seed(123456)

directory = "simulations"
heat_values = []
ks = []
sims = 199
for path in os.listdir(directory):
    full_path = os.path.join(directory, path)
    if os.path.isfile(full_path) and 'stone' in full_path:
        file = np.load(full_path)
        mean_heat = 0
        fn, fn2, k = path.replace('.npz', '').split('_')
        ks.append(float(k))
        print("k=" + k)

        for sim in range(sims):
            filter_radius = 1
            file_name = f"{fn} {fn2}"
            index_map = file['indexMap'][sim, :, :]
            simulation = file['sim'][sim, :, :]
            simulation_size = index_map.shape[0] * index_map.shape[1]
            ti = file['ti']
            original_size = ti.shape[0] * ti.shape[1]

            heat_map_creator = VerbatimHeatMapCreator(index_map, simulation)
            heat_map = heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, inv_dist_weight_exp)
            mean_heat += round(HeatMapAnalysis(heat_map).mean_heat_value(), 10)
        heat_values.append(mean_heat / sims)

plt.scatter(ks, heat_values, color="red")
plt.xlabel('K')
plt.ylabel('Mean heat value')
plt.title(f'Relation of k to mean heat value averaged over {sims} simulations')
plt.savefig('output/Relation of k to mean heat value.png', dpi=150)
plt.show()
