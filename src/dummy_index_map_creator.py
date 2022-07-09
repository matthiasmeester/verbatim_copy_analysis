import random
from random import randint
from random import randrange
from random import shuffle

import numpy as np
import matplotlib.pyplot as plt


class DummyIndexMapCreator:

    def __init__(self, map_shape):
        self.map_shape = map_shape
        self.source_index = np.stack(
            np.meshgrid(np.arange(self.map_shape[0]) / self.map_shape[0],
                        np.arange(self.map_shape[1]) / self.map_shape[1]) + [
                np.ones(self.map_shape)], axis=-1)

    def create_checkerboard_map(self, square_size, plot=False):
        full_random = self.create_full_random_map()
        full_verbatim = self.create_full_verbatim_map()
        n_rows, n_columns = full_random.shape[0], full_random.shape[1]
        n_rows_, n_columns_ = int(n_rows / square_size + 1), int(n_columns / square_size + 1)
        rows_grid, columns_grid = np.meshgrid(range(n_rows_), range(n_columns_), indexing='ij')
        high_res_checkerboard = np.mod(rows_grid, 2) + np.mod(columns_grid, 2) == 1
        square = np.ones((square_size, square_size))
        checkerboard = np.kron(high_res_checkerboard, square)[:n_rows, :n_columns]
        checkerboard = np.where(checkerboard, full_verbatim, full_random).astype(np.int32)

        if plot:
            plt.imshow(np.reshape(self.source_index, (-1, 3))[checkerboard])
            plt.title('10x10 Checkerboard dummy index map')
            plt.axis('off')
            plt.savefig('output/checkerboard_index_map', dpi=150)
            plt.show()

        return checkerboard

    def create_patch_map(self, num_patches, noise=0.05, plot=False):
        full_random = self.create_full_random_map()
        full_verbatim = self.create_full_verbatim_map()
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
                    result[iy, ix] = full_verbatim[iy + shift_y, ix + shift_x]
                    is_verbatim[iy, ix] = True

        verbatim_copy_proportion = is_verbatim.sum() / total_size

        if plot:
            plt.imshow(np.reshape(self.source_index, (-1, 3))[result])
            plt.title(f'Patch dummy index map, proportion verbatim = {verbatim_copy_proportion}')
            plt.axis('off')
            plt.savefig('output/patch_index_map', dpi=150)
            plt.show()

        return result, verbatim_copy_proportion

    def create_long_range_map(self, proportion, plot=False):
        result = self.create_full_random_map()
        full_verbatim = self.create_full_verbatim_map()
        is_verbatim = np.full(result.shape, False)
        group_index = 0
        pixels_left_in_group = 0
        total_size = is_verbatim.shape[0] * is_verbatim.shape[1]
        indices = []

        for iy in range(result.shape[0]):
            for ix in range(result.shape[1]):
                indices.append((iy, ix))
        shuffle(indices)

        for (iy, ix) in indices:
            if pixels_left_in_group == 0:
                pixels_left_in_group = randint(20, 100)
                group_index += total_size
            if random.random() < proportion:
                is_verbatim[iy, ix] = True
                result[iy, ix] = full_verbatim[iy, ix] + group_index
            pixels_left_in_group -= 1

        verbatim_copy_proportion = is_verbatim.sum() / total_size

        if plot:
            plt.imshow(np.reshape(self.source_index, (-1, 3))[result])
            plt.title(f'Long range index map, proportion verbatim = {round(verbatim_copy_proportion, 3)}')
            plt.axis('off')
            plt.savefig('output/long_range_index_map', dpi=150)
            plt.show()

        return result, verbatim_copy_proportion

    def create_full_verbatim_map(self):
        total_size = self.map_shape[0] * self.map_shape[1]
        return np.arange(0, total_size).reshape(self.map_shape)

    def create_full_random_map(self):
        total_size = self.map_shape[0] * self.map_shape[1]
        return (np.random.rand(self.map_shape[0], self.map_shape[1]) * total_size).astype(np.int32)
