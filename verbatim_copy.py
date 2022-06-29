import os.path
from random import randrange

import matplotlib.pyplot as plt
import numpy as np

from src.dummy_index_map_creator import DummyIndexMapCreator
from src.heat_map_analysis import HeatMapAnalysis
from src.verbatim_heat_map_creator import VerbatimHeatMapCreator

# ----- Verbatim copy statistic: -----
# --- Custom variables ---
inv_dist_weight_exp = 1

# seed(123456)
# ---


index_map_creator = DummyIndexMapCreator((200, 200))

directory = "simulations"
for path in os.listdir(directory):
    full_path = os.path.join(directory, path)
    if os.path.isfile(full_path):
        file = np.load(full_path)
        n = 170
        verbatim_indices = []
        filter_radi = range(1, 51)
        filter_radi = [1]
        fn, fn2, k = path.replace('.npz', '').split('_')
        file_name = f"{fn} {fn2}"
        index_map = file['indexMap'][n, :, :]
        simulation = file['sim'][n, :, :]
        simulation_size = index_map.shape[0] * index_map.shape[1]
        ti = file['ti']
        original_size = ti.shape[0] * ti.shape[1]
        sourceIndex = np.stack(
            np.meshgrid(np.arange(ti.shape[0]) / ti.shape[0], np.arange(ti.shape[1]) / ti.shape[1]) + [
                np.ones_like(ti)], axis=-1)

        # index_map, patch_percentage = index_map_creator.create_patch_map(10, plot=True)
        # index_map = index_map_creator.create_full_random_map()
        # index_map, patch_percentage = index_map_creator.create_long_range_map(0.10)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6, 6))
        fig.suptitle(f'QS \'stone\', k={k}, n={n}', size='xx-large')
        ax1.imshow(ti)
        ax1.set_title('Training image')
        ax1.axis('off')
        ax2.imshow(simulation)
        ax2.set_title('Simulation')
        ax2.axis('off')

        ax3.imshow(sourceIndex)
        ax3.set_title('Training image index map')
        ax3.axis('off')
        ax4.imshow(np.reshape(sourceIndex, (-1, 3))[index_map])
        ax4.set_title('Simulation index map')
        ax4.axis('off')
        plt.savefig('output/input_example', dpi=150)
        plt.show()

        heat_map_creator = VerbatimHeatMapCreator(index_map)

        for filter_radius in filter_radi:
            # --- Do simulations ---
            heat_map = heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, inv_dist_weight_exp)
            non_weighted_heat_map = heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, 0)

            nw_max_heat_threshold = 0.00246278367601897
            # _, nw_max_heat_threshold = heat_map_creator.noise_heat_statistics(index_map.shape, filter_radius, 0)
            noise_scalar = 100
            nw_max_heat_threshold *= noise_scalar
            heat_map_including_neighbours = \
               heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, inv_dist_weight_exp, 2, 1)
            long_range_heat_map = \
                heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius, 1, inverse_distance_weighted=True)

            # --- Calculate statistics ---
            dist_weighted_mean_heat_value = HeatMapAnalysis(heat_map).mean_heat_value()
            non_weighted_mean_heat_value = HeatMapAnalysis(non_weighted_heat_map).mean_heat_value()
            # long_range_mean_heat_value = round(HeatMapAnalysis(long_range_heat_map).mean_heat_value(), 4)
            verbatim_indices.append(dist_weighted_mean_heat_value)
            proportion_above_stat = HeatMapAnalysis(non_weighted_heat_map).above_treshold_heat_proportion(nw_max_heat_threshold)
            proportion_above_0_001 = HeatMapAnalysis(non_weighted_heat_map).above_treshold_heat_proportion(0.001)
            proportion_above_0_5 = HeatMapAnalysis(non_weighted_heat_map).above_treshold_heat_proportion(0.5)
            # mean_heat_value_with_neighbours = round(HeatMapAnalysis(heat_map_including_neighbours).mean_heat_value(), 4)
            patch_number, largest_patch_size, mean_patch_size = HeatMapAnalysis(non_weighted_heat_map).patch_stats(
                heat_treshold=nw_max_heat_threshold,
                patch_size_treshold=5,
                plot=True
            )

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
            ax1.imshow(np.reshape(sourceIndex, (-1, 3))[index_map], interpolation='none')
            ax1.set_title(f'Simulation index map')
            ax1.axis('off')
            ax2.imshow(heat_map, interpolation='none')
            ax2.set_title(f'Verbatim heat map, $r={filter_radius}$, $d={inv_dist_weight_exp}$')
            ax2.axis('off')
            heat_map_im = ax3.imshow(heat_map_including_neighbours, interpolation='none')
            ax3.set_title(f'Neighbour verbatim heat map,\n $r={filter_radius}$, $d={inv_dist_weight_exp}$, $r_{{cl}}={2}$, $d_{{cl}}={1}$')
            ax3.axis('off')
            fig.colorbar(heat_map_im, ax=ax3)
            plt.savefig(f'output/heat_map_example.png', bbox_inches='tight', dpi=150)
            plt.show()

            print(f"--- Filter_radius: {filter_radius} ---")
            print(f"Global statistics:")
            # print(f"Verbatim occurs on average with distance: {round(mean_verbatim_dist, 2)}")
            print(f"Short range statistics:")
            print(f"Proportion of pixels >= statistically likely of neighbours being verbatim: {proportion_above_stat}")
            print(f"Proportion of pixels >= 0.001 of neighbours being verbatim: {proportion_above_0_001}")
            print(f"Proportion of pixels >= 0.5 of neighbours being verbatim: {proportion_above_0_5}")
            print(f"Inverse distance weighted mean heat value: {dist_weighted_mean_heat_value}")
            # print(f"Factor verbatim copy over noise verbatim copy: {non_weighted_mean_heat_value / non_weighted_normalizer}")
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
            ax3.set_title(f'MHV={round(dist_weighted_mean_heat_value, 3)}, PROP={round(proportion_above_stat, 3)}')
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
