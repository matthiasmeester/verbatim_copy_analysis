import os.path
from io import BytesIO
from random import randrange

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from g2s import g2s

from VerbatimAnalysis import VerbatimAnalysis

np.set_printoptions(precision=3)

# load example training image ('stone')
img_w, img_h = 200, 200
url = 'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff'
# url = 'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/concrete.tiff'
ti = np.array(Image.open(BytesIO(requests.get(url).content)))

index_map_fp = 'index_map.txt'
simulation_fp = 'simulation.txt'
new_simulation = False
if os.path.isfile(index_map_fp) and os.path.isfile(simulation_fp) and not new_simulation:
    index_map = np.loadtxt(index_map_fp, dtype='int')
    simulation = np.loadtxt(simulation_fp)
else:
    # QS call using G2S
    simulation, index_map, _ = g2s('-a', 'qs',
                                   '-ti', ti,
                                   '-di', np.zeros((200, 200)) * np.nan,
                                   '-dt', [0],  # Zero for continuous variables
                                   '-k', 1.2,
                                   '-n', 50,
                                   '-j', 0.5)
    np.savetxt('index_map.txt', index_map, fmt='%i')
    np.savetxt('simulation.txt', simulation)


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
            ti = file['ti']
            # simulation = ti.copy()
            # index_map = np.arange(0, 40000).reshape((200, 200))
            verbatim_analysis = VerbatimAnalysis(index_map, simulation)
            similarity_map = verbatim_analysis.get_short_range_verbatim_heat_map(filter_radius, inv_dist_weight_exp)
            mean_heat_value = verbatim_analysis.mean_heat_value(similarity_map)
            verbatim_indices.append(mean_heat_value)
            # plt.hist(list(similarity_map.reshape(40000)), bins=100)
            # plt.show()

            # smoothed_map = verbatim_analysis.smooth_similarity_map(similarity_map, smoothing_radius, smoothing_exp)

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

            # keep values above treshold
            # av_map = similarity_map.copy()
            # treshold = 0.1
            # av_map[av_map < treshold] = 0

            #
            non_weighted_sim_map = verbatim_analysis.get_short_range_verbatim_heat_map(filter_radius, 0)
            proportion_above_0_25 = verbatim_analysis.above_treshold_heat_index(non_weighted_sim_map, 0.25)
            proportion_above_0_5 = verbatim_analysis.above_treshold_heat_index(non_weighted_sim_map, 0.5)
            proportion_above_0_75 = verbatim_analysis.above_treshold_heat_index(non_weighted_sim_map, 0.75)
            proportion_above_1_0 = verbatim_analysis.above_treshold_heat_index(non_weighted_sim_map, 1.0)

            print(f"--- Filter_radius: {filter_radius} ---")
            print(f"Short range statistics:")
            print(f"prop_lt_0_25 = {proportion_above_0_25}")
            print(f"prop_lt_0_50 = {proportion_above_0_5}")
            print(f"prop_lt_0_75 = {proportion_above_0_75}")
            print(f"prop_lt_1_00 = {proportion_above_1_0}")
            print(f"mean_heat_value = {round(mean_heat_value, 4)}")
            print("---\n")

            # alt_verbatim_indices.append(alt_mean_heat_value)
            #
            # av_img = ax4.imshow(av_map, interpolation='none')
            # ax4.set_title(f'av={alt_mean_heat_value}')
            # ax4.axis('off')
            # fig.colorbar(av_img)
            plt.show()
            np.savetxt('similarity_map.txt', similarity_map)

        plt.scatter(filter_radi, verbatim_indices, color="red")
        plt.scatter(filter_radi, alt_verbatim_indices, color="blue")
        plt.xlabel('Filter radius')
        plt.ylabel('Verbatim index')
        plt.title(f'Filter radius to verbatim index - {file_name}, k={k}, d={inv_dist_weight_exp}')
        plt.show()
