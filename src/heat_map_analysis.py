import cv2
import matplotlib.pyplot as plt
import numpy as np


class HeatMapAnalysis:

    def __init__(self, heat_map):
        self.heat_map = heat_map

    def mean_heat_value(self):
        return np.mean(self.heat_map)

    def above_treshold_heat_index(self, threshold=0.7):
        return np.sum(self.heat_map >= threshold) / self.heat_map.size

    def patch_stats(self, heat_treshold=1.0, patch_size_treshold=1, plot=False):
        # Todo smooth heat map with a self made function that takes edges in to account
        # smoothed_heat_map = VerbatimHeatMapCreator.smooth_heat_map(self.heat_map, 2, 1)
        filtered_heat_map = np.where(self.heat_map >= heat_treshold, 1, 0)
        pixel_map = (np.array(filtered_heat_map * 255)).astype(np.uint8)
        rgb_pixel_map = cv2.cvtColor(pixel_map, cv2.COLOR_GRAY2RGB)

        ret, thresh = cv2.threshold(pixel_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

        largest_patch_size = 0
        for i in range(1, n_labels):
            patch_area = stats[i, cv2.CC_STAT_AREA]
            if patch_area >= patch_size_treshold:
                largest_patch_size = max(patch_area, largest_patch_size)
                if plot:
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    cv2.rectangle(rgb_pixel_map, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
        if plot:
            plt.figure()
            plt.title(f'Verbatim patches with size >= {patch_size_treshold} pix, heat_treshold >= {heat_treshold}')
            plt.axis('off')
            plt.imshow(rgb_pixel_map)
        return n_labels - 1, largest_patch_size
