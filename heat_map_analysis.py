import cv2
import matplotlib.pyplot as plt
import numpy as np

from verbatim_heat_map_creator import VerbatimHeatMapCreator


class HeatMapAnalysis:

    def __init__(self, heat_map):
        self.heat_map = heat_map

    def mean_heat_value(self):
        return np.mean(self.heat_map)

    def above_treshold_heat_index(self, threshold=0.7):
        return np.sum(self.heat_map >= threshold) / self.heat_map.size

    def patch_stats(self, heat_treshold=0.5, patch_size_treshold=1, plot=False):
        smoothed_heat_map = VerbatimHeatMapCreator.smooth_heat_map(self.heat_map, 2, 1)
        filtered_heat_map = np.where(smoothed_heat_map > heat_treshold, 1, 0)
        pixel_map = (np.array(filtered_heat_map * 255)).astype(np.uint8)
        rgb_pixel_map = cv2.cvtColor(pixel_map, cv2.COLOR_GRAY2RGB)

        contours, hierarchy = cv2.findContours(pixel_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        patch_sizes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            patch_size = w * h
            if patch_size > patch_size_treshold:
                # Todo patch size should only count pixel values and not be square
                patch_sizes.append(patch_size)
                if plot:
                    cv2.rectangle(rgb_pixel_map, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)

        if plot:
            plt.figure()
            plt.imshow(rgb_pixel_map)

        number_of_boxes, largest_box_size = len(patch_sizes), max(patch_sizes)

        return number_of_boxes, largest_box_size
