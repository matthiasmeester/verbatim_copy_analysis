from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.heat_map_analysis import HeatMapAnalysis
from src.verbatim_heat_map_creator import VerbatimHeatMapCreator


def get_cel_color(val):
    if val == 0:
        return ""
    return f"\\cellcolor[gray]{{{1 - val}}}"


#  --- Variables ---
max_range = int(sqrt(200 ** 2 + 200 * 2))
threshold_range = 3
# _, max_noise_heat = VerbatimHeatMapCreator.noise_heat_statistics((200, 200), max_range, 0)
# table_str +=max_noise_heat)
max_noise_heat_s = 0.00246278367601897
max_noise_heat_l = 0.00010004966211932229
filter_radius = 1
inv_dist_weight_exp = 1
sim_type = 'strebelle'

ns = [1, 5, 10, 50, 100, 199]
ks = ['1.0', '1.5', '2.5', '3.0', '10.0']

for sim_type in ['stone', 'strebelle']:

    # ns = [1, 5]
    # ks = ['1.0', '1.5']

    n_columns = ''.join([f'& \\textbf{{$n={n}$}}' for n in ns])
    column_defs = ''.join(['l|' for _ in range(len(ns) + 1)])
    table_str = "\\begin{table}[ht]\n" \
                "\\centering\n" \
                f"\\begin{{tabular}}{{|{column_defs}}}\n" \
                "\\hline\n" \
                f"\\textbf{{}} {n_columns}\\\\ \\hline\n"

    for k in tqdm(ks):
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
            index_map = file['indexMap'][n - 1, :, :]
            simulation = file['sim'][n - 1, :, :]
            simulation_size = index_map.shape[0] * index_map.shape[1]

            # --- Do filter analysis ---
            heat_map_creator = VerbatimHeatMapCreator(index_map)
            heat_map = heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, inv_dist_weight_exp)
            non_weighted_heat_map_s = heat_map_creator.get_verbatim_heat_map_filter_basis(threshold_range, 0)
            non_weighted_heat_map_l = heat_map_creator.get_verbatim_heat_map_filter_basis(max_range, 0)

            # --- Calculate statistics ---
            dist_weighted_mean_heat_value = HeatMapAnalysis(heat_map).mean_heat_value()
            proportion_above_stat_s = HeatMapAnalysis(non_weighted_heat_map_s).above_treshold_heat_proportion(100 * max_noise_heat_s)
            proportion_above_stat_l = HeatMapAnalysis(non_weighted_heat_map_l).above_treshold_heat_proportion(100 * max_noise_heat_l)

            # fig = plt.figure(figsize=(5, 5))
            plt.imshow(np.reshape(sourceIndex, (-1, 3))[index_map], interpolation='none')
            plt.axis('off')
            plt.savefig(f'output/{sim_type}_table/index_maps/k{k.replace(".", "x")}n{n}.png', bbox_inches='tight', dpi=50)
            plt.clf()
            plt.imshow(simulation,interpolation='none')
            plt.axis('off')
            plt.savefig(f'output/{sim_type}_table/simulation_maps/k{k.replace(".", "x")}n{n}.png', bbox_inches='tight', dpi=50)
            plt.clf()
            mhvs.append(dist_weighted_mean_heat_value)
            propls.append(proportion_above_stat_l)
            propss.append(proportion_above_stat_s)
            row_images.append(f'sections/results/figures/{sim_type}_table/index_maps/k{k.replace(".", "x")}n{n}')

        # Create table rows
        table_str += f"$k={k}$"
        for row_images in row_images:
            table_str += f" & \\rowincludegraphics[scale=0.17]{{{row_images}.png}}"
        table_str += "\\\\ \\hline\n"

        table_str += f"$MHV$"
        for val in mhvs:
            table_str += f" & {round(val, 3)}"
        table_str += "\\\\ \\hline\n"

        table_str += f"$TNA_l$"
        for val in propls:
            table_str += f" & {round(val, 3)}"
        table_str += "\\\\ \\hline\n"

        table_str += f"$TNA_s$"
        for val in propss:
            table_str += f" & {round(val, 3)}"
        table_str += "\\\\ \\hline\n"

    table_str += " \n\\end{tabular}" \
                 "\\caption{\\label{tab: " + f"'{sim_type}'" + " simulation index map results}Statistics on various types of " + f"'{sim_type}'" + \
                 " simulations, with varying $n$ and $k$. " \
                 "Showing the $MHV$, $TNA_l$ and $TNA_s$ and the simulation index map.}" \
                 "\n\\end{table}\n"

    with open(f"output/{sim_type}_table/table_index_maps.tex", "w") as text_file:
        text_file.write(table_str.replace('scale=0.16', 'scale=0.2'))

    with open(f"output/{sim_type}_table/table_simulation_maps.tex", "w") as text_file:
        text_file.write(table_str.replace('index_maps', 'simulation_maps').replace('simulation index map', 'simulation'))
