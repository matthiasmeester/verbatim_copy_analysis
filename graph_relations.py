import os.path
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.heat_map_analysis import HeatMapAnalysis
from src.verbatim_heat_map_creator import VerbatimHeatMapCreator

# File for generating plots, very messy.

max_range = int(sqrt(200 ** 2 + 200 * 2))

inv_dist_weight_exp = 2
directory = "simulations"
k_heat_values = []
k_props_100mnh_s = []
k_props_100mnh_l = []
k_max_patch_sizes = []
k_mean_patch_sizes = []
ks = []
ns = list(range(0, 199))
ns.sort()
print(ns)

n_heat_values = [0] * len(ns)
n_props_100mnh_s = [0] * len(ns)
n_props_100mnh_l = [0] * len(ns)
n_max_patch_sizes = [0] * len(ns)
n_mean_patch_sizes = [0] * len(ns)

max_noise_heat_l = 0.00010004966211932229
max_noise_heat_s = 0.00246278367601897

for path in tqdm(os.listdir(directory)):
    full_path = os.path.join(directory, path)
    if os.path.isfile(full_path) and 'stone' in full_path:
        file = np.load(full_path)
        mean_heat = 0
        mean_100mnh_s = 0
        mean_100mnh_l = 0
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
            non_weighted_heat_map_r_3 = heat_map_creator.get_verbatim_heat_map_filter_basis(3, 0)
            non_weighted_heat_map_r_max = heat_map_creator.get_verbatim_heat_map_filter_basis(max_range, 0)
            heat_map = heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, inv_dist_weight_exp)
            heat = round(HeatMapAnalysis(heat_map).mean_heat_value(), 10)

            max_range_heat_map = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius=max_range, inv_dist_weight_exp=0))

            _, max_patch_size, mean_patch_size = HeatMapAnalysis(non_weighted_heat_map_r_3).patch_stats(
                heat_treshold=100 * max_noise_heat_s,
                patch_size_treshold=5,
                plot=False
            )
            max_patch_size /= simulation_size
            mean_patch_size /= simulation_size

            proportion_above_stat_s = HeatMapAnalysis(non_weighted_heat_map_r_3).above_treshold_heat_index(100 * max_noise_heat_s)
            proportion_above_stat_l = HeatMapAnalysis(non_weighted_heat_map_r_max).above_treshold_heat_index(100 * max_noise_heat_l)

            mean_heat += heat
            mean_100mnh_s += proportion_above_stat_s
            mean_100mnh_l += proportion_above_stat_l
            mean_mean_cs += mean_patch_size
            mean_max_cs += max_patch_size

            n_heat_values[i] += heat
            n_props_100mnh_s[i] += proportion_above_stat_s
            n_props_100mnh_l[i] += proportion_above_stat_l
            n_mean_patch_sizes[i] += mean_patch_size
            n_max_patch_sizes[i] += max_patch_size

        k_heat_values.append(mean_heat / len(ns))
        k_props_100mnh_s.append(mean_100mnh_s / len(ns))
        k_props_100mnh_l.append(mean_100mnh_l / len(ns))
        k_mean_patch_sizes.append(mean_mean_cs / len(ns))
        k_max_patch_sizes.append(mean_max_cs / len(ns))

n_heat_values = [x / len(ks) for x in n_heat_values]
n_props_100mnh_s = [x / len(ks) for x in n_props_100mnh_s]
n_props_100mnh_l = [x / len(ks) for x in n_props_100mnh_l]
n_max_patch_sizes = [x / len(ks) for x in n_max_patch_sizes]
n_mean_patch_sizes = [x / len(ks) for x in n_mean_patch_sizes]

print('ks', ks)
print('heat_values', k_heat_values)
print('props_10mnh', k_props_100mnh_s)
print('props_100mnh', k_props_100mnh_l)
print('k_max_patch_sizes', k_max_patch_sizes)
print('k_mean_patch_sizes', k_mean_patch_sizes)

print('ns', ns)
print('n_data_heat_values', n_heat_values)
print('n_data_props_10mnh', n_props_100mnh_s)
print('n_data_props_100mnh', n_props_100mnh_l)
print('n_max_patch_sizes', n_max_patch_sizes)
print('n_mean_patch_sizes', n_mean_patch_sizes)


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(ks, k_heat_values, color="b", s=2, label="MHV")
ax1.scatter(ks, k_props_100mnh_s, color="r", s=2, label="PNA s")
ax1.scatter(ks, k_props_100mnh_l, color="g", s=2, label="PNA l")
plt.xlabel('k')
plt.legend(loc='upper right')
plt.ylabel('Value')
plt.title(f'Relation of k to statistics')
plt.savefig('output/relations/Relation of k to statistics', dpi=150)
plt.show()

plt.scatter(ks, k_mean_patch_sizes, color="b", s=2)
plt.xlabel('k')
plt.ylabel('Patch proportion')
plt.title(f'Relation of k to mean patch size')
plt.savefig('output/relations/Relation of k to mean patch size.png', dpi=150)
plt.show()

plt.scatter(ks, k_max_patch_sizes, color="b", s=2)
plt.xlabel('k')
plt.ylabel('Patch proportion')
plt.title(f'Relation of k to max patch size')
plt.savefig('output/relations/Relation of k to max patch size.png', dpi=150)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(ns, n_heat_values, color="b", s=2, label="MHV")
ax1.scatter(ns, n_props_100mnh_s, color="r", s=2, label="PNA s")
ax1.scatter(ns, n_props_100mnh_l, color="g", s=2, label="PNA l")
plt.xlabel('n')
plt.legend(loc='upper left')
plt.ylabel('Value')
plt.title(f'Relation of n to statistics')
plt.savefig('output/relations/Relation of n to statistics', dpi=150)
plt.show()

plt.scatter(ns, n_mean_patch_sizes, color="b", s=2)
plt.xlabel('n')
plt.ylabel('Patch proportion')
plt.title(f'Relation of n to mean patch size')
plt.savefig('output/relations/Relation of n to mean patch size.png', dpi=150)
plt.show()

plt.scatter(ns, n_max_patch_sizes, color="b", s=2)
plt.xlabel('n')
plt.ylabel('Patch proportion')
plt.title(f'Relation of n to max patch size')
plt.savefig('output/relations/Relation of n to max patch size.png', dpi=150)
plt.show()
