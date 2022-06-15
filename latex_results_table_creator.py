from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from src.heat_map_analysis import HeatMapAnalysis
from src.verbatim_heat_map_creator import VerbatimHeatMapCreator

#  --- Variables ---
max_range = int(sqrt(200 ** 2 + 200 * 2))
threshold_range = 3
# _, max_noise_heat = VerbatimHeatMapCreator.noise_heat_statistics((200, 200), max_range, 0)
# table_str +=max_noise_heat)
max_noise_heat_s = 0.00246278367601897
max_noise_heat_l = 0.00010004966211932229
filter_radius = 1
inv_dist_weight_exp = 1
sim_type = 'stone'

ns = [1, 5, 10, 50, 100, 198]
ks = ['1.0', '1.5', '2.5', '3.0', '10.0']

# ns = [1, 5]
# ks = ['1.0', '1.5']


n_columns = ''.join([f'& \\textbf{{n={n}}}' for n in ns])
column_defs = ''.join(['l|' for _ in range(len(ns) + 1)])
table_str = "\\newcommand\\rowincludegraphics[2][]{\\raisebox{-0.45\\height}{\\includegraphics[#1]{#2}}}\n" \
            "\\begin{table}[t]\n" \
            f"\\begin{{tabular}}{{|{column_defs}}}\n" \
            "\\hline\n" \
            f"\\textbf{{}} {n_columns}\\\\ \\hline\n"

for k in ks:
    mhvs = []
    propls = []
    propss = []
    row_images = []

    file = np.load(f'simulations/qsSim_{sim_type}_{k}.npz')
    ti = file['ti']
    original_size = ti.shape[0] * ti.shape[1]
    sourceIndex = np.stack(
        np.meshgrid(np.arange(ti.shape[0]) / ti.shape[0], np.arange(ti.shape[1]) / ti.shape[1]) + [
            np.ones_like(ti)], axis=-1)

    for n in ns:
        index_map = file['indexMap'][n, :, :]
        simulation = file['sim'][n, :, :]
        simulation_size = index_map.shape[0] * index_map.shape[1]

        # --- Do filter analysis ---
        heat_map_creator = VerbatimHeatMapCreator(index_map)
        heat_map = heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, inv_dist_weight_exp)
        non_weighted_heat_map_s = heat_map_creator.get_verbatim_heat_map_filter_basis(threshold_range, 0)
        non_weighted_heat_map_l = heat_map_creator.get_verbatim_heat_map_filter_basis(max_range, 0)

        # --- Calculate statistics ---
        dist_weighted_mean_heat_value = HeatMapAnalysis(heat_map).mean_heat_value()
        proportion_above_stat_s = HeatMapAnalysis(non_weighted_heat_map_s).above_treshold_heat_index(100 * max_noise_heat_s)
        proportion_above_stat_l = HeatMapAnalysis(non_weighted_heat_map_l).above_treshold_heat_index(100 * max_noise_heat_l)

        # fig = plt.figure(figsize=(5, 5))
        plt.imshow(np.reshape(sourceIndex, (-1, 3))[index_map])
        plt.axis('off')
        plt.savefig(f'output/table/k{k.replace(".", "x")}n{n}.png', bbox_inches='tight', dpi=150)
        plt.clf()
        mhvs.append(dist_weighted_mean_heat_value)
        propls.append(proportion_above_stat_l)
        propss.append(proportion_above_stat_s)
        row_images.append(f'sections/results/figures/table/k{k.replace(".", "x")}n{n}')

    # Create table rows
    table_str += f"k={k}"
    for row_images in row_images:
        table_str += f" & \\rowincludegraphics[scale=0.2]{{{row_images}.png}}"
    table_str += "\\\\ \\hline\n"

    table_str += f"MHV"
    for val in mhvs:
        table_str += f" & {round(val, 3)}"
    table_str += "\\\\ \\hline\n"

    table_str += f"PNA l"
    for val in propls:
        table_str += f" & {round(val, 3)}"
    table_str += "\\\\ \\hline\n"

    table_str += f"PNA s"
    for val in propss:
        table_str += f" & {round(val, 3)}"
    table_str += "\\\\ \\hline\n"

table_str += " \n\\end{tabular}" \
            "\\caption{\\label{tab:Simulation results}Statistics on various types of simulations, with varying $n$ and $k$.}" \
             "\n\\end{table}\n"
with open("output/table/table.tex", "w") as text_file:
    text_file.write(table_str)