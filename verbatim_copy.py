import os.path
from random import randrange

import matplotlib.pyplot as plt
import numpy as np

from src.dummy_index_map_creator import DummyIndexMapCreator
from src.heat_map_analysis import HeatMapAnalysis
from src.verbatim_heat_map_creator import VerbatimHeatMapCreator

# ----- Verbatim copy statistic: -----
# --- Custom variables ---
inv_dist_weight_exp = 2

# seed(123456)
# ---

# From: https://stackoverflow.com/questions/38083788/turn-grid-into-a-checkerboard-pattern-in-python

index_map_creator = DummyIndexMapCreator((200, 200))

directory = "simulations"
for path in os.listdir(directory):
    full_path = os.path.join(directory, path)
    if os.path.isfile(full_path):
        file = np.load(full_path)
        random_index = randrange(199)
        verbatim_indices = []
        filter_radi = range(1, 51)
        filter_radi = [400]
        fn, fn2, k = path.replace('.npz', '').split('_')
        file_name = f"{fn} {fn2}"
        index_map = file['indexMap'][random_index, :, :]
        simulation = file['sim'][random_index, :, :]
        simulation_size = index_map.shape[0] * index_map.shape[1]
        ti = file['ti']
        original_size = ti.shape[0] * ti.shape[1]
        sourceIndex = np.stack(
            np.meshgrid(np.arange(ti.shape[0]) / ti.shape[0], np.arange(ti.shape[1]) / ti.shape[1]) + [
                np.ones_like(ti)], axis=-1)

        index_map = index_map_creator.create_checkerboard_map(3)

        heat_map_creator = VerbatimHeatMapCreator(index_map, simulation)
        n_fr = 70
        neighbourhood_verbatim, distances = heat_map_creator.neighbourhood_verbatim_analysis(
            filter_radius=n_fr)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle(f'NVA - {file_name}', size='xx-large')
        ax1.set_title('Probability verbatim at distance histogram')
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Probability')
        ax1.scatter(list(distances.keys()), list(distances.values()))
        # ax1.plot(range(40), [math.sqrt(2 * x ** 2) for x in range(40)], 'r--')
        neigh_img = ax2.imshow(neighbourhood_verbatim, extent=[-n_fr, n_fr, -n_fr, n_fr])
        fig.colorbar(neigh_img, ax=ax2)
        ax2.set_title('Probability verbatim at distance map')
        ax2.set_xlabel('X distance')
        ax2.set_ylabel('Y distance')
        plt.show()

        for filter_radius in filter_radi:
            # --- Do simulations ---
            heat_map = heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, inv_dist_weight_exp)
            non_weighted_heat_map = heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, 0)
            # heat_map_including_neighbours = \
            #    heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, inv_dist_weight_exp, 2, 1)
            long_range_heat_map = \
                heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, 1, inverse_distance_weighted=True)

            # --- Calculate statistics ---
            dist_weighted_mean_heat_value = HeatMapAnalysis(heat_map).mean_heat_value()
            non_weighted_mean_heat_value = HeatMapAnalysis(non_weighted_heat_map).mean_heat_value()
            # long_range_mean_heat_value = round(HeatMapAnalysis(long_range_heat_map).mean_heat_value(), 4)
            verbatim_indices.append(dist_weighted_mean_heat_value)
            proportion_above_0_001 = HeatMapAnalysis(non_weighted_heat_map).above_treshold_heat_index(0.001)
            proportion_above_0_5 = HeatMapAnalysis(non_weighted_heat_map).above_treshold_heat_index(0.5)
            proportion_above_1_0 = HeatMapAnalysis(non_weighted_heat_map).above_treshold_heat_index(1.0)
            # mean_heat_value_with_neighbours = round(HeatMapAnalysis(heat_map_including_neighbours).mean_heat_value(), 4)
            patch_number, largest_patch_size = HeatMapAnalysis(non_weighted_heat_map).patch_stats(
                heat_treshold=0.001,
                patch_size_treshold=5,
                plot=True)

            print(f"--- Filter_radius: {filter_radius} ---")
            print(f"Global statistics:")
            # print(f"Verbatim occurs on average with distance: {round(mean_verbatim_dist, 2)}")
            print(f"Short range statistics:")
            print(f"Proportion of pixels >= 0.01% of neighbours being verbatim: {proportion_above_0_001}")
            print(f"Proportion of pixels >= 50% of neighbours being verbatim: {proportion_above_0_5}")
            print(f"Proportion of pixels >= 100% of neighbours being verbatim: {proportion_above_1_0}")
            print(f"Inverse distance weighted mean heat value: {dist_weighted_mean_heat_value}")
            print(f"Non weighted mean heat value: {non_weighted_mean_heat_value}")
            # print(f"Inversely weighted mean heat value: {long_range_mean_heat_value}")
            # print(f"Mean heat value including close by verbatim: {mean_heat_value_with_neighbours}")
            print(f"Number of patches: {patch_number}")
            print(
                f"Largest continuous patch size: {largest_patch_size} pix, proportion sim: {largest_patch_size / simulation_size}, original: {largest_patch_size / original_size}")
            print("---\n")

            #  --- Do plotting ---
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))
            fig.suptitle(f'{file_name}, k={k}, r={filter_radius}, d={inv_dist_weight_exp}', size='xx-large')
            ax1.imshow(ti)
            ax1.set_title('Training image')
            ax1.axis('off')
            ax2.imshow(simulation)
            ax2.set_title('Simulation')
            # ax2.imshow(long_range_heat_map)
            # ax2.set_title(f'LR mean heat value={long_range_mean_heat_value}')
            ax2.axis('off')
            sim_img = ax3.imshow(heat_map, interpolation='none')
            ax3.set_title(f'Mean heat value={round(dist_weighted_mean_heat_value, 4)}')
            ax3.axis('off')
            fig.colorbar(sim_img, ax=ax3)
            plt.show()

            # plt.hist(list(heat_map.reshape(40000)), bins=100)
            # plt.show()

        plt.scatter(filter_radi, verbatim_indices, color="red")
        plt.xlabel('Filter radius')
        plt.ylabel('Verbatim index')
        plt.title(f'Filter radius to verbatim index - {file_name}, k={k}, d={inv_dist_weight_exp}')
        plt.show()
