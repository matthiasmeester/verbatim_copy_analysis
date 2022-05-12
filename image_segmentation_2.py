import math

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.ndimage
from sklearn.cluster import KMeans

similarity_map = np.loadtxt('similarity_map.txt')


def create_inv_weight_matrix(filter_radius, exponent, middle_weight=0):
    filter_range = math.ceil(filter_radius)
    weight_adj_matrix = np.zeros((filter_range * 2 + 1, filter_range * 2 + 1))
    for i, dx in enumerate(range(-filter_range, filter_range + 1)):
        for j, dy in enumerate(range(-filter_range, filter_range + 1)):
            if dx == 0 and dy == 0:
                weight_adj_matrix[j][i] = middle_weight
            else:
                # Inv weight adj
                distance = math.sqrt(dx ** 2 + dy ** 2)
                if distance <= filter_radius:
                    weight_adj_matrix[j][i] = math.pow(math.sqrt(dx ** 2 + dy ** 2), -exponent)
    return weight_adj_matrix


smooth_weights = create_inv_weight_matrix(2, 1, middle_weight=1)
similarity_map = sp.ndimage.filters.convolve(similarity_map, smooth_weights / np.sum(smooth_weights),
                                             mode='constant')
similarity_map[similarity_map < 0.8] = 0
similarity_map[similarity_map >= 0.8] = 1

sim_img = plt.imshow(similarity_map, interpolation='none')
plt.axis('off')
plt.colorbar(sim_img)
plt.show()


# k-means cluster similarity map
def cluster(similarity_map):
    similarity_map = np.reshape(similarity_map, (-1, 3))
    similarity_map = similarity_map[np.where(np.sum(similarity_map, axis=1) > 0)]
    similarity_map = np.reshape(similarity_map, (-1, 3))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(similarity_map)
