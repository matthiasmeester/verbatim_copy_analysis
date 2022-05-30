import os.path
import random
from random import randint
from random import randrange

import matplotlib.pyplot as plt
import numpy as np

from heat_map_analysis import HeatMapAnalysis
from verbatim_heat_map_creator import VerbatimHeatMapCreator

# ----- Verbatim copy statistic: -----
# --- Custom variables ---
inv_dist_weight_exp = 2


# seed(123456)
# ---

# From: https://stackoverflow.com/questions/38083788/turn-grid-into-a-checkerboard-pattern-in-python
def make_checkerboard(full_copy, full_random, square_size):
    n_rows, n_columns = full_random.shape[0], full_random.shape[1]
    n_rows_, n_columns_ = int(n_rows / square_size + 1), int(n_columns / square_size + 1)
    rows_grid, columns_grid = np.meshgrid(range(n_rows_), range(n_columns_), indexing='ij')
    high_res_checkerboard = np.mod(rows_grid, 2) + np.mod(columns_grid, 2) == 1
    square = np.ones((square_size, square_size))
    checkerboard = np.kron(high_res_checkerboard, square)[:n_rows, :n_columns]
    checkerboard = np.where(checkerboard, full_copy, full_random)
    return checkerboard.astype(np.int32)


def put_patches(full_copy, full_random, num_patches, patch_radius, noise=0.05):
    result = full_random.copy()
    is_verbatim = np.full(result.shape, False)
    total_size = is_verbatim.shape[0] * is_verbatim.shape[1]
    lr, rr = 5, 25
    for _ in range(num_patches):
        h, w = full_random.shape[:2]
        circle_pos_x, circle_pos_y = randrange(w), randrange(h)
        circle_pos_copy_x, circle_pos_copy_y = randrange(2 * rr, w - 2 * rr), randrange(2 * rr, h - 2 * rr)
        shift_x = circle_pos_copy_x - circle_pos_x
        shift_y = circle_pos_copy_y - circle_pos_y
        circle_r = randint(5, 25)

        for iy, ix in np.ndindex(result.shape):
            # Check if point is on circle and put some noise
            if (ix - circle_pos_x) ** 2 + (iy - circle_pos_y) ** 2 <= circle_r ** 2 and random.random() >= noise:
                result[iy, ix] = full_copy[iy + shift_y, ix + shift_x]
                is_verbatim[iy, ix] = True

    verbatim_copy_proportion = is_verbatim.sum() / total_size
    return result, verbatim_copy_proportion


def randomly_draw(base, to_sample_from, proportion):
    result = base.copy()
    is_verbatim = np.full(result.shape, False)
    total_size = is_verbatim.shape[0] * is_verbatim.shape[1]

    for iy, ix in np.ndindex(result.shape):
        if random.random() < proportion:
            is_verbatim[iy, ix] = True
            result[iy, ix] = to_sample_from[iy, ix]

    verbatim_copy_proportion = is_verbatim.sum() / total_size
    return result, verbatim_copy_proportion


directory = "simulations"
for path in os.listdir(directory):
    full_path = os.path.join(directory, path)
    if os.path.isfile(full_path):
        file = np.load(full_path)
        random_index = randrange(199)
        verbatim_indices = []
        filter_radi = range(1, 51)
        # filter_radi = [1]
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

        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6, 6))
        # fig.suptitle(f'QS stone, k={k}', size='xx-large')
        # ax1.imshow(ti)
        # ax1.set_title('Training image')
        # ax1.axis('off')
        # ax2.imshow(simulation)
        # ax2.set_title('Simulation')
        # ax2.axis('off')
        #
        # ax3.imshow(sourceIndex)
        # ax3.set_title('Training image index map')
        # ax3.axis('off')
        # ax4.imshow(np.reshape(sourceIndex, (-1, 3))[index_map])
        # ax4.set_title('Simulation index map')
        # ax4.axis('off')
        # plt.savefig('output/input_example_index_map', dpi=150)
        # plt.savefig('output/input_example', dpi=150)
        # plt.show()
        #
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
        # ax1.imshow(sourceIndex)
        # ax1.set_title('Training image')
        # ax1.axis('off')
        # ax2.imshow(np.reshape(sourceIndex, (-1, 3))[index_map])
        # ax2.set_title('Simulation')
        # ax2.axis('off')
        # plt.savefig('output/input_example_index_map', dpi=150)
        # plt.show()

        # verbatim copy:
        # simulation = ti.copy()
        # index_map = np.arange(0, 40000).reshape((200, 200))

        # Patches
        # index_map, verbatim_copy_proportion = put_patches(np.arange(0, 40000).reshape((200, 200)),
        #                                                   (np.random.rand(200, 200) * 40000).astype(np.int32), 10, 30)
        # plt.imshow(np.reshape(sourceIndex, (-1, 3))[index_map])
        # plt.title(f'Patch dummy index map, proportion verbatim = {round(verbatim_copy_proportion, 3)}')
        # plt.axis('off')
        # plt.savefig('output/patch_index_map', dpi=150)
        # plt.show()

        # Long range
        chance = 0.4
        index_map, verbatim_copy_proportion = randomly_draw((np.random.rand(200, 200) * 40000).astype(np.int32),
                                                            np.arange(0, 40000).reshape((200, 200)), chance)
        plt.imshow(np.reshape(sourceIndex, (-1, 3))[index_map])
        plt.title(f'Long range index map, proportion verbatim = {round(verbatim_copy_proportion, 3)}')
        plt.axis('off')
        plt.savefig('output/long_range_index_map', dpi=150)
        plt.show()

        # Randomness
        # index_map = (np.random.rand(200, 200) * 40000).astype(np.int32)
        # plt.imshow(np.reshape(sourceIndex, (-1, 3))[index_map])
        # plt.title('Random dummy index map')
        # plt.axis('off')
        # plt.savefig('output/random_index_map', dpi=150)
        # plt.show()

        # Checkerboard
        # index_map = make_checkerboard(np.arange(0, 40000).reshape((200, 200)),
        #                               (np.random.rand(200, 200) * 40000), 10)
        #
        # plt.imshow(np.reshape(sourceIndex, (-1, 3))[index_map])
        # plt.title('10x10 Checkerboard dummy index map')
        # plt.axis('off')
        # plt.savefig('output/checkerboard_index_map', dpi=150)
        # plt.show()

        # shuffled randomness:
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
            patch_number, largest_patch_size = HeatMapAnalysis(non_weighted_sim_map).patch_stats(patch_size_treshold=10,
                                                                                                 plot=True)

            print(f"--- Filter_radius: {filter_radius} ---")
            print(f"Global statistics:")
            # print(f"Verbatim occurs on average with distance: {round(mean_verbatim_dist, 2)}")
            print(f"Short range statistics:")
            print(f"Proportion of pixels >= 50% of neighbours being verbatim: {proportion_above_0_5}")
            print(f"Proportion of pixels >= 100% of neighbours being verbatim: {proportion_above_1_0}")
            print(f"Mean heat value: {mean_heat_value}")
            # print(f"Inversely weighted mean heat value: {long_range_mean_heat_value}")
            print(f"Mean heat value including close by verbatim: {mean_heat_value_with_neighbours}")
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
