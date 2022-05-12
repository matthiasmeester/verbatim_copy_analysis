from io import BytesIO

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth

mpl.rcParams['figure.dpi'] = 300

# Load in data
ti = np.array(Image.open(BytesIO(requests.get(
    'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff').content)))
similarity_map = np.loadtxt('similarity_map.txt')
treshold = 0.7
similarity_map[similarity_map < treshold] = 0
index_map = np.loadtxt('index_map.txt')
simulation = np.loadtxt('simulation.txt')
# Cluster

to_cluster = []
for iy, ix in np.ndindex(similarity_map.shape):
    similarity_value = similarity_map[iy][ix]
    to_cluster.append([iy, ix, similarity_value])
to_cluster = np.asarray(to_cluster)

bandwidth = estimate_bandwidth(to_cluster, quantile=0.001, n_samples=2000)
clustered = MeanShift(bandwidth=bandwidth * 3, bin_seeding=True).fit(to_cluster)
labels = clustered.labels_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
fig.suptitle('QS Unconditional simulation', size='xx-large')
ax1.imshow(ti)
ax1.set_title('Training image')
ax1.axis('off')
ax2.imshow(simulation)
ax2.set_title('Simulation')
ax2.axis('off')


def coord_to_index(x, y):
    return x + 200 * y


def index_to_coord(index):
    index = int(index)
    import math
    y = math.floor(index / 200)
    x = index - y * 200
    return int(x), int(y)


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


clustered_image = np.zeros((40000,))

i = 0
for iy, ix in np.ndindex(similarity_map.shape):
    coord_x, coord_y = index_to_coord(i)
    index = coord_to_index(ix, iy)
    i += 1

cmap = get_cmap(n_clusters_)
for i in range(n_clusters_):
    indices = np.where(clustered.labels_ == i)
    cluster_center = clustered.cluster_centers_[i]
    cluster_mean = np.mean(to_cluster[indices, 2])
    clustered_image[indices] = cluster_mean
    if cluster_mean > 0.7:
        # points_train = []
        # points_sim = []
        #
        # for x, y in zip(to_cluster[indices, 1].flatten(), to_cluster[indices, 0].flatten()):
        #     original_index = index_map[int(y)][int(x)]
        #     original_x, original_y = index_to_coord(original_index)
        #     points_sim.append((x, y))
        #     points_train.append((original_x, original_y))
        x = cluster_center[1]
        y = cluster_center[0]
        original_index = index_map[int(y)][int(x)]
        original_x, original_y = index_to_coord(original_index)
        ax1.add_patch(plt.Circle((original_x, original_y), 5, color=cmap(i), fill=False, alpha=0.9))
        ax2.add_patch(plt.Circle((x, y), 5, color=cmap(i), fill=False, alpha=0.9))

        # ax1.add_patch(Polygon(points_train, color=cmap(i), alpha=0.5))
        # ax2.add_patch(Polygon(points_sim, color=cmap(i), alpha=0.5))

plt.savefig('Circles.png')
plt.show()

plt.imshow(clustered_image.reshape((200, 200, 1)), interpolation='none')
plt.colorbar()
plt.show()
