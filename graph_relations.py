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
k_max_patch_sizes = []
k_mean_patch_sizes = []
ks = []
ns = list(range(0, 199, 40))

n_heat_values = [0] * len(ns)
n_props_10mnh = [0] * len(ns)
n_props_100mnh = [0] * len(ns)
n_max_patch_sizes = [0] * len(ns)
n_mean_patch_sizes = [0] * len(ns)

max_range = int(sqrt(200 ** 2 + 200 * 2))
_, max_noise_heat = VerbatimHeatMapCreator.noise_heat_statistics((200, 200), max_range, 0)

for path in tqdm(os.listdir(directory)):
    full_path = os.path.join(directory, path)
    if os.path.isfile(full_path) and 'stone' in full_path:
        file = np.load(full_path)
        mean_heat = 0
        mean_10mnh = 0
        mean_100mnh = 0
        mean_mean_cs = 0
        mean_max_cs = 0
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
            non_weighted_heat_map = heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, 0)
            heat_map = heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, inv_dist_weight_exp)
            heat = round(HeatMapAnalysis(heat_map).mean_heat_value(), 10)
            max_range_heat_map = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius=max_range, inv_dist_weight_exp=0))

            prop_10mnh = max_range_heat_map.above_treshold_heat_index(10 * max_noise_heat)
            prop_100mnh = max_range_heat_map.above_treshold_heat_index(100 * max_noise_heat)

            _, max_patch_size, mean_patch_size = HeatMapAnalysis(non_weighted_heat_map).patch_stats(
                heat_treshold=100 * max_noise_heat,
                patch_size_treshold=5,
                plot=False
            )
            max_patch_size /= simulation_size
            mean_patch_size /= simulation_size

            mean_heat += heat
            mean_10mnh += prop_10mnh
            mean_100mnh += prop_100mnh
            mean_mean_cs += mean_patch_size
            mean_max_cs += max_patch_size

            n_heat_values[i] += heat
            n_props_10mnh[i] += prop_10mnh
            n_props_100mnh[i] += prop_100mnh
            n_mean_patch_sizes[i] += mean_patch_size
            n_max_patch_sizes[i] += max_patch_size

        k_heat_values.append(mean_heat / len(ns))
        k_props_10mnh.append(mean_10mnh / len(ns))
        k_props_100mnh.append(mean_100mnh / len(ns))
        k_mean_patch_sizes.append(mean_mean_cs / len(ns))
        k_max_patch_sizes.append(mean_max_cs / len(ns))

n_heat_values = [x / len(ks) for x in n_heat_values]
n_props_10mnh = [x / len(ks) for x in n_props_10mnh]
n_props_100mnh = [x / len(ks) for x in n_props_100mnh]
n_max_patch_sizes = [x / len(ks) for x in n_max_patch_sizes]
n_mean_patch_sizes = [x / len(ks) for x in n_mean_patch_sizes]

print('ks', ks)
print('heat_values', k_heat_values)
print('props_10mnh', k_props_10mnh)
print('props_100mnh', k_props_100mnh)
print('k_max_patch_sizes', k_max_patch_sizes)
print('k_mean_patch_sizes', k_mean_patch_sizes)

print('ns', ns)
print('n_data_heat_values', n_heat_values)
print('n_data_props_10mnh', n_props_10mnh)
print('n_data_props_100mnh', n_props_100mnh)
print('n_max_patch_sizes', n_max_patch_sizes)
print('n_mean_patch_sizes', n_mean_patch_sizes)

plt.scatter(ks, k_heat_values, color="b")
plt.xlabel('k')
plt.ylabel('Mean heat value')
plt.title(f'Relation of k to MHV')
plt.savefig('output/relations/Relation of k to MHV', dpi=150)
plt.show()

plt.scatter(ks, k_props_10mnh, color="b")
plt.xlabel('k')
plt.ylabel('Proportion of pixels >=T')
plt.title(f'Relation of k to PNA, T=10mnh')
plt.savefig('output/relations/Relation of k to PNA, T=10mnh.png', dpi=150)
plt.show()

plt.scatter(ks, k_props_100mnh, color="b")
plt.xlabel('k')
plt.ylabel('Proportion of pixels >=T')
plt.title(f'Relation of k to PNA, T=100mnh')
plt.savefig('output/relations/Relation of k to PNA, T=100mnh.png', dpi=150)
plt.show()

plt.scatter(ks, k_mean_patch_sizes, color="b")
plt.xlabel('k')
plt.ylabel('Patch proportion')
plt.title(f'Relation of k to mean patch size')
plt.savefig('output/relations/Relation of k to mean patch size.png', dpi=150)
plt.show()

plt.scatter(ks, k_max_patch_sizes, color="b")
plt.xlabel('k')
plt.ylabel('Patch proportion')
plt.title(f'Relation of k to max patch size')
plt.savefig('output/relations/Relation of k to max patch size.png', dpi=150)
plt.show()

plt.scatter(ns, n_heat_values, color="b")
plt.xlabel('n')
plt.ylabel('Mean heat value')
plt.title(f'Relation of n to MHV')
plt.savefig('output/relations/Relation of n to MHV.png', dpi=150)
plt.show()

plt.scatter(ns, n_props_10mnh, color="b")
plt.xlabel('n')
plt.ylabel('Proportion of pixels >=T')
plt.title(f'Relation of n to PNA, T=10mnh')
plt.savefig('output/relations/Relation of n to PNA, T=10mnh.png', dpi=150)
plt.show()

plt.scatter(ns, n_props_100mnh, color="b")
plt.xlabel('n')
plt.ylabel('Proportion of pixels >=T')
plt.title(f'Relation of n to PNA, T=100mnh')
plt.savefig('output/relations/Relation of n to PNA, T=100mnh.png', dpi=150)
plt.show()

plt.scatter(ns, n_mean_patch_sizes, color="b")
plt.xlabel('n')
plt.ylabel('Patch proportion')
plt.title(f'Relation of n to mean patch size')
plt.savefig('output/relations/Relation of n to mean patch size.png', dpi=150)
plt.show()

plt.scatter(ns, n_max_patch_sizes, color="b")
plt.xlabel('n')
plt.ylabel('Patch proportion')
plt.title(f'Relation of n to max patch size')
plt.savefig('output/relations/Relation of n to max patch size.png', dpi=150)
plt.show()
