import os.path
from io import BytesIO
from random import randrange

import matplotlib.pyplot as plt
from VerbatimAnalysis import VerbatimAnalysis
import numpy as np
import requests
from PIL import Image
from g2s import g2s

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
inv_dist_weight_exp = 1
smoothing_radius = 0
smoothing_exp = 2
# ---

# verbatim_analysis = VerbatimAnalysis(index_map, simulation, ti)
# verbatim_index, similarity_map = verbatim_analysis.filter_verbatim_statistic(filter_radius, inv_dist_weight_exp,
#                                                                              smoothing_radius, smoothing_exp)
# print("Verbatim index: " + verbatim_index)
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
        filter_radi = range(1, 52)
        fn, fn2, k = path.replace('.npz', '').split('_')
        file_name = f"{fn} {fn2}"

        for filter_radius in filter_radi:
            index_map = file['indexMap'][random_index, :, :]
            simulation = file['sim'][random_index, :, :]
            ti = file['ti']
            # simulation = ti.copy()
            # index_map = np.arange(0, 40000).reshape((200, 200))
            verbatim_analysis = VerbatimAnalysis(index_map, simulation)
            verbatim_index, similarity_map = \
                verbatim_analysis.filter_verbatim_statistic(filter_radius, inv_dist_weight_exp, smoothing_radius,
                                                            smoothing_exp)
            verbatim_indices.append(verbatim_index)
            print(filter_radius)

            # Plotting
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))
            fig.suptitle(f'{file_name}, k={k}, r={filter_radius}, d={inv_dist_weight_exp}',
                         size='xx-large')
            ax1.imshow(ti)
            ax1.set_title('Training image')
            ax1.axis('off')
            ax2.imshow(simulation)
            ax2.set_title('Simulation')
            ax2.axis('off')

            sim_img = ax3.imshow(similarity_map, interpolation='none')
            ax3.set_title(f'Verbatim Map, v={round(verbatim_index, 4)}')
            ax3.axis('off')
            fig.colorbar(sim_img)
            plt.show()

        plt.scatter(filter_radi, verbatim_indices, color="red")
        plt.xlabel('Filter radius')
        plt.ylabel('Verbatim index')
        plt.title(f'Filter radius to verbatim index - {file_name}, k={k}, d={inv_dist_weight_exp}')
        plt.show()
