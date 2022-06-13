import os.path
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.heat_map_analysis import HeatMapAnalysis
from src.verbatim_heat_map_creator import VerbatimHeatMapCreator

# ----- Verbatim copy statistic: -----
# --- Custom variables ---
inv_dist_weight_exp = 2

directory = "simulations"
heat_values = []
props_10mnh = []
props_100mnh = []
ks = []
sims = 199
max_range = int(sqrt(200 ** 2 + 200 * 2))
_, max_noise_heat = VerbatimHeatMapCreator.noise_heat_statistics((200, 200), max_range, 0)
for path in os.listdir(directory):
    full_path = os.path.join(directory, path)
    if os.path.isfile(full_path) and 'stone' in full_path:
        file = np.load(full_path)
        mean_heat = 0
        mean_10mnh = 0
        mean_100mnh = 0
        fn, fn2, k = path.replace('.npz', '').split('_')
        ks.append(float(k))
        print("k=" + k)

        for sim in tqdm(range(sims)):
            filter_radius = 1
            file_name = f"{fn} {fn2}"
            index_map = file['indexMap'][sim, :, :]
            simulation = file['sim'][sim, :, :]
            simulation_size = index_map.shape[0] * index_map.shape[1]
            ti = file['ti']
            original_size = ti.shape[0] * ti.shape[1]

            heat_map_creator = VerbatimHeatMapCreator(index_map)
            heat_map = heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, inv_dist_weight_exp)
            mean_heat += round(HeatMapAnalysis(heat_map).mean_heat_value(), 10)
            max_range_heat_map = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius=max_range, inv_dist_weight_exp=0))

            mean_10mnh += max_range_heat_map.above_treshold_heat_index(10 * max_noise_heat)
            mean_100mnh += max_range_heat_map.above_treshold_heat_index(100 * max_noise_heat)

        heat_values.append(mean_heat / sims)
        props_10mnh.append(mean_10mnh / sims)
        props_100mnh.append(mean_100mnh / sims)

print('ks', ks)
print('heat_values', heat_values)
print('props_10mnh', props_10mnh)
print('props_100mnh', props_100mnh)

plt.scatter(ks, heat_values, color="red")
plt.xlabel('K')
plt.ylabel('Mean heat value')
plt.title(f'Relation of k to MHV')
plt.savefig('output/Relation of k to mean heat value.png', dpi=150)
plt.show()

plt.scatter(ks, heat_values, color="red")
plt.xlabel('K')
plt.ylabel('Proportion of pixels >=T')
plt.title(f'Relation of k to PNA, T=10mnh')
plt.savefig('output/Relation of k to proportional neighbour analysis 10mnh.png', dpi=150)
plt.show()

plt.scatter(ks, heat_values, color="red")
plt.xlabel('K')
plt.ylabel('Proportion of pixels >=T')
plt.title(f'Relation of k to PNA, T=100mnh')
plt.savefig('output/Relation of k to proportional neighbour analysis 100mnh.png', dpi=150)
plt.show()
