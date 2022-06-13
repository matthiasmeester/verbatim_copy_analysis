import os.path
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.heat_map_analysis import HeatMapAnalysis
from src.verbatim_heat_map_creator import VerbatimHeatMapCreator

# File for generating plots, very messy.

inv_dist_weight_exp = 2
directory = "simulations"
k_heat_values = []
k_props_10mnh = []
k_props_100mnh = []
ks = []
ns = list(range(199))

n_heat_values = [0] * len(ns)
n_props_10mnh = [0] * len(ns)
n_props_100mnh = [0] * len(ns)

max_range = int(sqrt(200 ** 2 + 200 * 2))
_, max_noise_heat = VerbatimHeatMapCreator.noise_heat_statistics((200, 200), max_range, 0)

for path in tqdm(os.listdir(directory)):
    full_path = os.path.join(directory, path)
    if os.path.isfile(full_path) and 'stone' in full_path:
        file = np.load(full_path)
        mean_heat = 0
        mean_10mnh = 0
        mean_100mnh = 0
        fn, fn2, k = path.replace('.npz', '').split('_')
        ks.append(float(k))

        for i, n in enumerate(ns):
            filter_radius = 1
            file_name = f"{fn} {fn2}"
            index_map = file['indexMap'][n, :, :]
            simulation = file['sim'][n, :, :]
            simulation_size = index_map.shape[0] * index_map.shape[1]
            ti = file['ti']
            original_size = ti.shape[0] * ti.shape[1]

            heat_map_creator = VerbatimHeatMapCreator(index_map)
            heat_map = heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, inv_dist_weight_exp)
            heat = round(HeatMapAnalysis(heat_map).mean_heat_value(), 10)
            max_range_heat_map = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius=max_range, inv_dist_weight_exp=0))

            prop_10mnh = max_range_heat_map.above_treshold_heat_index(10 * max_noise_heat)
            prop_100mnh = max_range_heat_map.above_treshold_heat_index(100 * max_noise_heat)

            mean_heat += heat
            mean_10mnh += prop_10mnh
            mean_100mnh += prop_100mnh

            n_heat_values[i] += heat
            n_props_10mnh[i] += prop_10mnh
            n_props_100mnh[i] += prop_100mnh

        k_heat_values.append(mean_heat / len(ns))
        k_props_10mnh.append(mean_10mnh / len(ns))
        k_props_100mnh.append(mean_100mnh / len(ns))

n_heat_values = [x / len(ks) for x in n_heat_values]
n_props_10mnh = [x / len(ks) for x in n_props_10mnh]
n_props_100mnh = [x / len(ks) for x in n_props_100mnh]

print('ks', ks)
print('heat_values', k_heat_values)
print('props_10mnh', k_props_10mnh)
print('props_100mnh', k_props_100mnh)

print('ns', ns)
print('n_data_heat_values', n_heat_values)
print('n_data_props_10mnh', n_props_10mnh)
print('n_data_props_100mnh', n_props_100mnh)

plt.scatter(ks, k_heat_values, color="red")
plt.xlabel('k')
plt.ylabel('Mean heat value')
plt.title(f'Relation of k to MHV')
plt.savefig('output/relations/Relation of k to MHV', dpi=150)
plt.show()

plt.scatter(ks, k_props_10mnh, color="red")
plt.xlabel('k')
plt.ylabel('Proportion of pixels >=T')
plt.title(f'Relation of k to PNA, T=10mnh')
plt.savefig('output/relations/Relation of k to PNA, T=10mnh.png', dpi=150)
plt.show()

plt.scatter(ks, k_props_100mnh, color="red")
plt.xlabel('k')
plt.ylabel('Proportion of pixels >=T')
plt.title(f'Relation of k to PNA, T=100mnh')
plt.savefig('output/relations/Relation of k to PNA, T=100mnh.png', dpi=150)
plt.show()

plt.scatter(ns, n_heat_values, color="red")
plt.xlabel('n')
plt.ylabel('Mean heat value')
plt.title(f'Relation of n to MHV')
plt.savefig('output/relations/Relation of n to MHV.png', dpi=150)
plt.show()

plt.scatter(ns, n_props_10mnh, color="red")
plt.xlabel('n')
plt.ylabel('Proportion of pixels >=T')
plt.title(f'Relation of n to PNA, T=10mnh')
plt.savefig('output/relations/Relation of n to PNA, T=10mnh.png', dpi=150)
plt.show()

plt.scatter(ns, n_props_100mnh, color="red")
plt.xlabel('n')
plt.ylabel('Proportion of pixels >=T')
plt.title(f'Relation of n to PNA, T=100mnh')
plt.savefig('output/relations/Relation of n to PNA, T=100mnh.png', dpi=150)
plt.show()
