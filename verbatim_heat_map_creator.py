import math
from functools import lru_cache

import numpy as np
import scipy as sp
import scipy.ndimage


class VerbatimHeatMapCreator:

    def __init__(self, index_map, simulation):
        self.index_map = index_map
        self.simulation = simulation
        assert(self.index_map.shape == self.simulation.shape)

    def get_short_range_verbatim_heat_map(self, filter_radius, inv_dist_weight_exp, include_neighbors_radius=0,
                                          neighbor_inv_dist_weight=1):
        # Create verbatim and inverse distance weight matrices
        heat_map = np.zeros((self.index_map.shape[0], self.index_map.shape[1]))
        weight_adj_matrix = self.create_inv_weight_matrix(filter_radius, inv_dist_weight_exp)

        @lru_cache(maxsize=2048)
        def sum_adj_weight_slice(wy0, wy1, wx0, wx1):
            return np.sum(weight_adj_matrix[wy0:wy1 + 1, wx0:wx1 + 1])

        # Verbatim adjacency matrix initialisation
        adj_matrix_size = filter_radius * 2 + 1
        verbatim_adj_matrix = np.zeros((adj_matrix_size, adj_matrix_size))
        for i, dx in enumerate(range(-filter_radius, filter_radius + 1)):
            for j, dy in enumerate(range(-filter_radius, filter_radius + 1)):
                verbatim_adj_matrix[j][i] = dx + self.index_map.shape[0] * dy

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

    def _index_to_coord(self, index):
        index = int(index)
        import math
        y = math.floor(index / self.index_map.shape[1])
        x = index - y * self.index_map.shape[1]
        return int(x), int(y)

    @staticmethod
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
