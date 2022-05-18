import os.path
from random import randrange

import matplotlib.pyplot as plt
import numpy as np

from heat_map_analysis import HeatMapAnalysis
from verbatim_heat_map_creator import VerbatimHeatMapCreator

np.set_printoptions(precision=6)

# ----- Verbatim copy statistic: -----
# --- Custom variables ---
inv_dist_weight_exp = 2
# seed(123456)
# ---

directory = "simulations"
for path in os.listdir(directory):
    full_path = os.path.join(directory, path)
    if os.path.isfile(full_path):
        file = np.load(full_path)
        random_index = randrange(199)
        verbatim_indices = []
        filter_radi = range(1, 51)
        fn, fn2, k = path.replace('.npz', '').split('_')
        file_name = f"{fn} {fn2}"
        index_map = file['indexMap'][random_index, :, :]
        simulation = file['sim'][random_index, :, :]
        image_size = index_map.shape[0] * index_map.shape[1]
        ti = file['ti']

        # Enable these lines to test full verbatim copy:
        # simulation = ti.copy()
        # index_map = np.arange(0, 40000).reshape((200, 200))

        # Enable these lines to test full randomness:
        # random_seed = np.random.randint(0, 1000)
        # np.random.seed(random_seed)
        # index_map = np.arange(0, 40000)
        # np.random.shuffle(index_map)
        # index_map = index_map.reshape((200, 200))
        # np.random.seed(random_seed)
        # simulation = simulation.reshape((40000, -1))
        # np.random.shuffle(simulation)
        # simulation = simulation.reshape((200, 200))

        # TODO: Test in between for example 50% verbatim copy with different patterns, chessboard for example

        heat_map_creator = VerbatimHeatMapCreator(index_map, simulation)
        n_fr = 60
        neighbourhood_verbatim, distance_verbatim_value_pairs = heat_map_creator.neighbourhood_verbatim_analysis(
            filter_radius=n_fr, min_filter_radius=0, normalize=False)
        distances = [x[0] for x in distance_verbatim_value_pairs]
        verbatim_values = [x[1] for x in distance_verbatim_value_pairs]
        mean_verbatim_dist = np.sum([x[0] * x[1] for x in distance_verbatim_value_pairs]) / np.sum(
            neighbourhood_verbatim)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle(f'NVA - {file_name}, mean_verbatim_dist={round(mean_verbatim_dist, 2)}', size='xx-large')
        ax1.set_title('Probability verbatim at distance histogram')
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Probability')
        ax1.bar(distances, verbatim_values)
        neigh_img = ax2.imshow(neighbourhood_verbatim, extent=[-n_fr, n_fr, -n_fr, n_fr])
        fig.colorbar(neigh_img, ax=ax2)
        ax2.set_title('Probability verbatim at distance map')
        ax2.set_xlabel('X distance')
        ax2.set_ylabel('Y distance')
        plt.show()

        for filter_radius in filter_radi:
            # --- Do simulations ---
            heat_map = heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, inv_dist_weight_exp)
            non_weighted_sim_map = heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, 0)
            heat_map_including_neighbours = \
                heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, inv_dist_weight_exp, 2, 1)
            long_range_heat_map = \
                heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, 1, inverse_distance_weighted=True)

            # --- Calculate statistics ---
            mean_heat_value = round(HeatMapAnalysis(heat_map).mean_heat_value(), 10)
            # long_range_mean_heat_value = round(HeatMapAnalysis(long_range_heat_map).mean_heat_value(), 4)
            verbatim_indices.append(mean_heat_value)
            proportion_above_0_5 = HeatMapAnalysis(non_weighted_sim_map).above_treshold_heat_index(0.5)
            proportion_above_1_0 = HeatMapAnalysis(non_weighted_sim_map).above_treshold_heat_index(1.0)
            mean_heat_value_with_neighbours = round(HeatMapAnalysis(heat_map_including_neighbours).mean_heat_value(), 4)
            patch_number, largest_box_size = HeatMapAnalysis(non_weighted_sim_map).patch_stats(patch_size_treshold=10,
                                                                                               plot=True)

            print(f"--- Filter_radius: {filter_radius} ---")
            print(f"Global statistics:")
            print(f"Verbatim occurs on average with distance: {round(mean_verbatim_dist, 2)}")
            print(f"Short range statistics:")
            print(f"Proportion of pixels >= 50% of neighbours being verbatim: {proportion_above_0_5}")
            print(f"Proportion of pixels >= 100% of neighbours being verbatim: {proportion_above_1_0}")
            print(f"Mean heat value: {mean_heat_value}")
            # print(f"Inversely weighted mean heat value: {long_range_mean_heat_value}")
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
            # ax2.imshow(long_range_heat_map)
            # ax2.set_title(f'LR mean heat value={long_range_mean_heat_value}')
            ax2.axis('off')
            sim_img = ax3.imshow(heat_map, interpolation='none')
            ax3.set_title(f'Mean heat value={round(mean_heat_value, 4)}')
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
