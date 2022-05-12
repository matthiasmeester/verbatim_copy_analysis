import os.path
from random import randrange

import matplotlib.pyplot as plt
import numpy as np

from VerbatimAnalysis import VerbatimAnalysis

np.set_printoptions(precision=3)


# --- Display results ---
def plot():
    sourceIndex = np.stack(
        np.meshgrid(np.arange(ti.shape[0]) / ti.shape[0], np.arange(ti.shape[1]) / ti.shape[1]) + [
            np.ones_like(ti)], axis=-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
    fig.suptitle('QS Unconditional simulation', size='xx-large')
    ax1.imshow(ti)
    ax1.set_title('Training image')
    ax1.axis('off')
    ax2.imshow(simulation)
    ax2.set_title('Simulation')
    ax2.axis('off')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Unconditional simulation index map')
    ax1.imshow(sourceIndex)
    ax1.set_title('Training image')
    ax1.axis('off')
    ax2.imshow(np.reshape(sourceIndex, (-1, 3))[index_map])
    ax2.set_title('Simulation')
    ax2.axis('off')
    plt.show()


# ----- Verbatim copy statistic: -----

# --- Custom variables ---
# filter_radius = 51
inv_dist_weight_exp = 2
smoothing_radius = 3
smoothing_exp = 2
# ---

# verbatim_analysis = VerbatimAnalysis(index_map, simulation, ti)
# mean_heat_value, similarity_map = verbatim_analysis.get_short_range_verbatim_heat_map(filter_radius, inv_dist_weight_exp,
#                                                                              smoothing_radius, smoothing_exp)
# print("Verbatim index: " + mean_heat_value)
# plt.imshow(similarity_map, interpolation='none')
# plt.title("Similarity Map")
# plt.colorbar()
# plt.show()


d = "simulations"
for path in os.listdir(d):
    full_path = os.path.join(d, path)
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
            # simulation = ti.copy()
            # index_map = np.arange(0, 40000).reshape((200, 200))
            verbatim_analysis = VerbatimAnalysis(index_map, simulation)
            similarity_map = verbatim_analysis.get_short_range_verbatim_heat_map(filter_radius, inv_dist_weight_exp)
            similarity_map_including_neighbours = verbatim_analysis.get_short_range_verbatim_heat_map(filter_radius,
                                                                                                      inv_dist_weight_exp,
                                                                                                      2, 1)
            mean_heat_value = round(verbatim_analysis.mean_heat_value(similarity_map), 4)
            verbatim_indices.append(mean_heat_value)
            # plt.hist(list(similarity_map.reshape(40000)), bins=100)
            # plt.show()

            # Plotting
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))
            fig.suptitle(f'{file_name}, k={k}, r={filter_radius}, d={inv_dist_weight_exp}', size='xx-large')
            ax1.imshow(ti)
            ax1.set_title('Training image')
            ax1.axis('off')
            ax2.imshow(simulation)
            ax2.set_title('Simulation')
            ax2.axis('off')

            sim_img = ax3.imshow(similarity_map, interpolation='none')
            ax3.set_title(f'v={round(mean_heat_value, 4)}')
            ax3.axis('off')
            fig.colorbar(sim_img, ax=ax3)

            #
            non_weighted_sim_map = verbatim_analysis.get_short_range_verbatim_heat_map(filter_radius, 0)
            proportion_above_0_5 = verbatim_analysis.above_treshold_heat_index(non_weighted_sim_map, 0.5)
            mean_heat_value_with_neighbours = round(
                verbatim_analysis.mean_heat_value(similarity_map_including_neighbours), 4)

            number_of_patches, largest_box_size = \
                verbatim_analysis.patch_stats(similarity_map, patch_size_treshold=10, plot=False)

            print(f"--- Filter_radius: {filter_radius} ---")
            print(f"Short range statistics:")
            print(f"Proportion of pixels with more than 50% of neighbours being verbatim: {proportion_above_0_5}")
            print(f"Mean heat value: {mean_heat_value}")
            print(f"Mean heat value including close by verbatim: {mean_heat_value_with_neighbours}")
            print(f"Number of patches: {number_of_patches}")
            print(f"Largest continuous patch size: {largest_box_size}, proportion: {largest_box_size / image_size}")
            print("---\n")

            plt.show()

        plt.scatter(filter_radi, verbatim_indices, color="red")
        plt.scatter(filter_radi, alt_verbatim_indices, color="blue")
        plt.xlabel('Filter radius')
        plt.ylabel('Verbatim index')
        plt.title(f'Filter radius to verbatim index - {file_name}, k={k}, d={inv_dist_weight_exp}')
        plt.show()
