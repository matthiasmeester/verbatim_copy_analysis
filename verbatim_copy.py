import os.path
from random import randrange

import matplotlib.pyplot as plt
import numpy as np

from heat_map_analysis import HeatMapAnalysis
from verbatim_heat_map_creator import VerbatimHeatMapCreator

np.set_printoptions(precision=3)

# ----- Verbatim copy statistic: -----
# --- Custom variables ---
# filter_radius = 51
inv_dist_weight_exp = 2
smoothing_radius = 3
smoothing_exp = 2
# ---

directory = "simulations"
for path in os.listdir(directory):
    full_path = os.path.join(directory, path)
    if os.path.isfile(full_path):
        file = np.load(full_path)
        random_index = randrange(200)
        verbatim_indices = []
        alt_verbatim_indices = []
        filter_radi = range(1, 51)
        fn, fn2, k = path.replace('.npz', '').split('_')
        file_name = f"{fn} {fn2}"

        for filter_radius in filter_radi:
            index_map = file['indexMap'][random_index, :, :]
            simulation = file['sim'][random_index, :, :]
            image_size = index_map.shape[0] * index_map.shape[1]
            ti = file['ti']
            # Enable these 2 lines to test full verbatim copy:
            # simulation = ti.copy()
            # index_map = np.arange(0, 40000).reshape((200, 200))

            # --- Do simulations ---
            heat_map_creator = VerbatimHeatMapCreator(index_map, simulation)
            heat_map = heat_map_creator.get_short_range_verbatim_heat_map(filter_radius, inv_dist_weight_exp)
            non_weighted_sim_map = heat_map_creator.get_short_range_verbatim_heat_map(filter_radius, 0)
            heat_map_including_neighbours = \
                heat_map_creator.get_short_range_verbatim_heat_map(filter_radius, inv_dist_weight_exp, 2, 1)

            # --- Calculate statistics ---
            mean_heat_value = round(HeatMapAnalysis(heat_map).mean_heat_value(), 4)
            verbatim_indices.append(mean_heat_value)
            proportion_above_0_5 = HeatMapAnalysis(non_weighted_sim_map).above_treshold_heat_index(0.5)
            mean_heat_value_with_neighbours = round(HeatMapAnalysis(heat_map_including_neighbours).mean_heat_value(), 4)
            patch_number, largest_box_size = HeatMapAnalysis(heat_map).patch_stats(patch_size_treshold=10, plot=False)

            print(f"--- Filter_radius: {filter_radius} ---")
            print(f"Short range statistics:")
            print(f"Proportion of pixels with more than 50% of neighbours being verbatim: {proportion_above_0_5}")
            print(f"Mean heat value: {mean_heat_value}")
            print(f"Mean heat value including close by verbatim: {mean_heat_value_with_neighbours}")
            print(f"Number of patches: {patch_number}")
            print(f"Largest continuous patch size: {largest_box_size} pix, proportion: {largest_box_size / image_size}")
            print("---\n")

            #  --- Do plotting ---
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))
            fig.suptitle(f'{file_name}, k={k}, r={filter_radius}, d={inv_dist_weight_exp}', size='xx-large')
            ax1.imshow(ti)
            ax1.set_title('Training image')
            ax1.axis('off')
            ax2.imshow(simulation)
            ax2.set_title('Simulation')
            ax2.axis('off')
            sim_img = ax3.imshow(heat_map, interpolation='none')
            ax3.set_title(f'v={round(mean_heat_value, 4)}')
            ax3.axis('off')
            fig.colorbar(sim_img, ax=ax3)
            plt.show()
            # plt.hist(list(heat_map.reshape(40000)), bins=100)
            # plt.show()

        plt.scatter(filter_radi, verbatim_indices, color="red")
        plt.scatter(filter_radi, alt_verbatim_indices, color="blue")
        plt.xlabel('Filter radius')
        plt.ylabel('Verbatim index')
        plt.title(f'Filter radius to verbatim index - {file_name}, k={k}, d={inv_dist_weight_exp}')
        plt.show()
