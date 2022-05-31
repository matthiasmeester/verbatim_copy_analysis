import math
from collections import defaultdict
from functools import lru_cache

import numpy as np
import scipy as sp
import scipy.ndimage


class VerbatimHeatMapCreator:

    def __init__(self, index_map, simulation):
        self.index_map = index_map.astype(np.int32)
        self.simulation = simulation
        assert (self.index_map.shape == self.simulation.shape)

    def get_verbatim_heat_map_filter_basis(self, filter_radius, inv_dist_weight_exp, include_neighbors_radius=0,
                                           neighbor_inv_dist_weight=1, inverse_distance_weighted=False):
        # Create verbatim and inverse distance weight matrices
        heat_map = np.zeros((self.index_map.shape[0], self.index_map.shape[1]))
        weight_adj_matrix = self.create_inv_weight_matrix(filter_radius, inv_dist_weight_exp, middle_weight=0)
        verbatim_adj_matrix = self._create_verbatim_adj_matrix(filter_radius)

        if inverse_distance_weighted:
            weight_adj_matrix = 1 - weight_adj_matrix

        @lru_cache(maxsize=2048)
        def sum_adj_weight_slice(wy0, wy1, wx0, wx1):
            return np.sum(weight_adj_matrix[wy0:wy1 + 1, wx0:wx1 + 1])

        # Loop over all pixels to check if there is any verbatim copy
        for iy, ix in np.ndindex(self.index_map.shape):
            current_index = self.index_map[iy][ix]
            x0, x1, y0, y1, wx0, wx1, wy0, wy1 = self._get_filter_bounds(filter_radius, ix, iy, self.index_map.shape)
            im_selection = self.index_map[y0:y1 + 1, x0:x1 + 1]
            verb_selection = verbatim_adj_matrix[wy0:wy1 + 1, wx0:wx1 + 1] + current_index
            weight_selection = weight_adj_matrix[wy0:wy1 + 1, wx0:wx1 + 1]
            if include_neighbors_radius > 0:
                equality_matrix = self._get_verbatim_distance(im_selection, verb_selection, include_neighbors_radius,
                                                              neighbor_inv_dist_weight)
            else:
                equality_matrix = np.equal(im_selection, verb_selection)
            similarity_adj_matrix = equality_matrix * weight_selection
            heat_map[iy][ix] = np.sum(similarity_adj_matrix) / sum_adj_weight_slice(wy0, wy1, wx0, wx1)

        sum_adj_weight_slice.cache_clear()
        return heat_map

    def neighbourhood_verbatim_analysis(self, filter_radius):
        neighbourhood_verbatim = np.zeros((filter_radius * 2 + 1, filter_radius * 2 + 1))
        verbatim_adj_matrix = self._create_verbatim_adj_matrix(filter_radius)
        normalization = np.zeros((filter_radius * 2 + 1, filter_radius * 2 + 1))

        # Loop over all pixels to check if there is any verbatim copy
        for iy, ix in np.ndindex(self.index_map.shape):
            current_index = self.index_map[iy][ix]
            x0, x1, y0, y1, wx0, wx1, wy0, wy1 = self._get_filter_bounds(filter_radius, ix, iy, self.index_map.shape)
            im_selection = self.index_map[y0:y1 + 1, x0:x1 + 1]
            verb_selection = verbatim_adj_matrix[wy0:wy1 + 1, wx0:wx1 + 1] + current_index
            neighbourhood_verbatim[wy0:wy1 + 1, wx0:wx1 + 1] += np.equal(im_selection, verb_selection)
            normalization[wy0:wy1 + 1, wx0:wx1 + 1] += 1

        neighbourhood_verbatim[filter_radius, filter_radius] = 0
        neighbourhood_verbatim /= normalization
        distances = defaultdict(list)

        # Calculate average verbatim distance
        for iy, ix in np.ndindex(neighbourhood_verbatim.shape):
            distance_to_center = int(math.sqrt((iy - filter_radius) ** 2 + (ix - filter_radius) ** 2))
            if neighbourhood_verbatim[iy][ix] != 0 and distance_to_center <= filter_radius:
                distances[distance_to_center].append(neighbourhood_verbatim[iy][ix])

        for key in distances.keys():
            distances[key] = np.mean(distances[key])

        return neighbourhood_verbatim, distances

    @staticmethod
    def _sigmoid(m):
        return 1 / (1 + np.exp(-m))

    def _create_verbatim_adj_matrix(self, filter_radius):
        # Verbatim adjacency matrix initialisation
        adj_matrix_size = filter_radius * 2 + 1
        verbatim_adj_matrix = np.zeros((adj_matrix_size, adj_matrix_size))
        for i, dx in enumerate(range(-filter_radius, filter_radius + 1)):
            for j, dy in enumerate(range(-filter_radius, filter_radius + 1)):
                verbatim_adj_matrix[j][i] = dx + self.index_map.shape[0] * dy

        verbatim_adj_matrix[filter_radius, filter_radius] = -1
        return verbatim_adj_matrix.astype(np.int32)

    def _get_verbatim_distance(self, im_selection, verb_selection, include_neighbors_radius, neighbor_inv_dist_weight):
        heat_map = np.zeros((im_selection.shape[0], im_selection.shape[1]))

        for iy, ix in np.ndindex(im_selection.shape):
            if im_selection[iy][ix] == verb_selection[iy][ix]:
                heat_map[iy][ix] = 1
            else:
                im_x, im_y = self._index_to_coord(im_selection[iy][ix])
                verb_x, verb_y = self._index_to_coord(verb_selection[iy][ix])
                distance = math.sqrt((im_x - verb_x) ** 2 + (im_y - verb_y) ** 2)
                if distance <= include_neighbors_radius:
                    heat_map[iy][ix] = math.pow(distance, -neighbor_inv_dist_weight)
        return heat_map

    @lru_cache(maxsize=4096)
    def _index_to_coord(self, index):
        index = int(index)
        import math
        y = math.floor(index / self.index_map.shape[1])
        x = index - y * self.index_map.shape[1]
        return int(x), int(y)

    @staticmethod
    def create_inv_weight_matrix(filter_radius, exponent, middle_weight=0, include_outside_filter_radius=False):
        filter_range = math.ceil(filter_radius)
        weight_adj_matrix = np.zeros((filter_range * 2 + 1, filter_range * 2 + 1))
        for i, dx in enumerate(range(-filter_range, filter_range + 1)):
            for j, dy in enumerate(range(-filter_range, filter_range + 1)):
                if dx == 0 and dy == 0:
                    weight_adj_matrix[j][i] = middle_weight
                else:
                    # Inv weight adj
                    distance = math.sqrt(dx ** 2 + dy ** 2)
                    if distance <= filter_radius or include_outside_filter_radius:
                        weight_adj_matrix[j][i] = math.pow(distance, -exponent)
        return weight_adj_matrix

    @staticmethod
    def _get_filter_bounds(filter_radius, ix, iy, index_map_shape):
        map_h, map_w = index_map_shape
        adj_matrix_size = 2 * filter_radius + 1
        # Slice the selection with clip
        x0, x1 = ix - filter_radius, ix + filter_radius
        y0, y1 = iy - filter_radius, iy + filter_radius
        wx0, wx1, wy0, wy1 = 0, adj_matrix_size, 0, adj_matrix_size

        # Set out of bound filter values to clip to the maximum values
        # Adjust the weight matrix bounds and verbatim copy matrix bounds accordingly as well.
        if x0 < 0:
            wx0 = 0 - x0
            x0 = 0
        if x1 > map_w - 1:
            wx1 = adj_matrix_size - x1 + map_w - 2
            x1 = map_w - 1
        if y0 < 0:
            wy0 = 0 - y0
            y0 = 0
        if y1 > map_h - 1:
            wy1 = adj_matrix_size - y1 + map_h - 2
            y1 = map_h - 1

        return x0, x1, y0, y1, wx0, wx1, wy0, wy1

    @staticmethod
    def smooth_heat_map(heat_map, smoothing_radius, smoothing_exp):
        smooth_weights = VerbatimHeatMapCreator.create_inv_weight_matrix(smoothing_radius, smoothing_exp,
                                                                         middle_weight=1)
        heat_map = sp.ndimage.filters.convolve(heat_map, smooth_weights / np.sum(smooth_weights), mode='constant')
        return heat_map
