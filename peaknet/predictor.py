#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pickle
import cupy as cp

from math import isnan
from cupyx.scipy import ndimage

from peaknet.datasets.transform import coord_crop_to_img


class CheetahPeakFinder:

    def __init__(self, model = None, path_cheetah_geom = None):
        self.model             = model
        self.path_cheetah_geom = path_cheetah_geom

        # Load the cheetah geometry...
        with open(self.path_cheetah_geom, 'rb') as handle:
            cheetah_geom_dict = pickle.load(handle)
        self.cheetah_geom_list = list(cheetah_geom_dict.values())[::2]


    def calc_batch_center_of_mass(self, batch_mask):
        batch_mask = cp.asarray(batch_mask)

        # Set up structure to find connected component in 2D only...
        structure = cp.zeros((3, 3, 3))
        #                     ^  ^^^^
        # batch_______________|   |
        #                         |
        # 2D image________________|

        # Define structure in 2D image at the middle layer
        structure[1] = cp.array([[1,1,1],
                                 [1,1,1],
                                 [1,1,1]])

        # Fetch labels...
        batch_label, batch_num_feature = ndimage.label(batch_mask, structure = structure)

        # Calculate batch center of mass...
        batch_center_of_mass = ndimage.center_of_mass(batch_mask, batch_label, cp.asarray(range(1, batch_num_feature+1)))

        return batch_center_of_mass


    def calc_batch_center_of_mass_perf(self, batch_mask):
        import time

        batch_mask = cp.asarray(batch_mask)

        # Set up structure to find connected component in 2D only...
        structure = cp.zeros((3, 3, 3))
        #                     ^  ^^^^
        # batch_______________|   |
        #                         |
        # 2D image________________|

        # Define structure in 2D image at the middle layer
        structure[1] = cp.array([[1,1,1],
                                 [1,1,1],
                                 [1,1,1]])

        # Fetch labels...
        time_start = time.time()
        batch_label, batch_num_feature = ndimage.label(batch_mask, structure = structure)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Center of mass(L)'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        # Calculate batch center of mass...
        time_start = time.time()
        batch_center_of_mass = ndimage.center_of_mass(batch_mask, batch_label, cp.asarray(range(1, batch_num_feature+1)))
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Center of mass(C)'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        return batch_center_of_mass


    def calc_batch_mean_position(self, batch_mask):
        batch_mask = cp.asarray(batch_mask)

        # Set up structure to find connected component in 2D only...
        structure = cp.zeros((3, 3, 3))
        #                     ^  ^^^^
        # batch_______________|   |
        #                         |
        # 2D image________________|

        # Define structure in 2D image at the middle layer
        structure[1] = cp.array([[1,1,1],
                                 [1,1,1],
                                 [1,1,1]])

        # Fetch labels...
        batch_label, batch_num_feature = ndimage.label(batch_mask, structure = structure)

        # Calculate batch center of mass...
        batch_mean_position = [ cp.argwhere(batch_label == i).mean(axis = 0) for i in range(1, batch_num_feature + 1) ]

        return batch_mean_position


    def find_peak(self, img_stack, threshold_prob = 1 - 1e-4, min_num_peaks = 15):
        peak_list = []

        # Normalize the image stack...
        img_stack = (img_stack - img_stack.mean(axis = (2, 3), keepdim = True)) / img_stack.std(axis = (2, 3), keepdim = True)

        # Get activation feature map given the image stack...
        self.model.eval()
        with torch.no_grad():
            fmap_stack = self.model.forward(img_stack)

        # Convert to probability with the sigmoid function...
        mask_stack_predicted = fmap_stack.sigmoid()

        # Thresholding the probability...
        ## is_background = mask_stack_predicted < threshold_prob
        ## mask_stack_predicted[ is_background ] = 0
        ## mask_stack_predicted[~is_background ] = 1
        mask_stack_predicted = (mask_stack_predicted >= threshold_prob).type(torch.int32)

        # Find center of mass for each image in the stack...
        num_stack, _, size_y, size_x = mask_stack_predicted.shape
        peak_pos_predicted_stack = self.calc_batch_center_of_mass(mask_stack_predicted.view(num_stack, size_y, size_x))

        # A workaround to avoid copying gpu memory to cpu when num of peaks is small...
        if len(peak_pos_predicted_stack) >= min_num_peaks:
            # Convert to cheetah coordinates...
            for peak_pos in peak_pos_predicted_stack:
                idx_panel, y, x = peak_pos.get()

                if isnan(y) or isnan(x): continue

                idx_panel = int(idx_panel)

                y, x = coord_crop_to_img((y, x), img_stack.shape[-2:], mask_stack_predicted.shape[-2:])

                x_min, y_min, x_max, y_max = self.cheetah_geom_list[idx_panel]

                x += x_min
                y += y_min

                peak_list.append((y, x))

        return peak_list


    def find_peak_and_perf(self, img_stack, threshold_prob = 1 - 1e-4, min_num_peaks = 15):
        import time

        peak_list = []

        time_start = time.time()
        # Normalize the image stack...
        img_stack = (img_stack - img_stack.mean(axis = (2, 3), keepdim = True)) / img_stack.std(axis = (2, 3), keepdim = True)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Normalization'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Get activation feature map given the image stack...
        self.model.eval()
        with torch.no_grad():
            fmap_stack = self.model.forward(img_stack)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Inference'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Convert to probability with the sigmoid function...
        mask_stack_predicted = fmap_stack.sigmoid()
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Sigmoid'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Thresholding the probability...
        ## is_background = mask_stack_predicted < threshold_prob
        ## mask_stack_predicted[ is_background ] = 0
        ## mask_stack_predicted[~is_background ] = 1

        mask_stack_predicted = (mask_stack_predicted >= threshold_prob).type(torch.int32)

        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Thresholding'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Find center of mass for each image in the stack...
        num_stack, _, size_y, size_x = mask_stack_predicted.shape
        ## peak_pos_predicted_stack = self.calc_batch_center_of_mass(mask_stack_predicted.view(num_stack, size_y, size_x))
        peak_pos_predicted_stack = self.calc_batch_center_of_mass_perf(mask_stack_predicted.view(num_stack, size_y, size_x))
        ## peak_pos_predicted_stack = self.calc_batch_mean_position(mask_stack_predicted.view(num_stack, size_y, size_x))
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Center of mass'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        # A workaround to avoid copying gpu memory to cpu when num of peaks is small...
        if len(peak_pos_predicted_stack) >= min_num_peaks:
            time_start = time.time()
            # Convert to cheetah coordinates...
            for peak_pos in peak_pos_predicted_stack:
                idx_panel, y, x = peak_pos.get()

                if isnan(y) or isnan(x): continue

                idx_panel = int(idx_panel)

                y, x = coord_crop_to_img((y, x), img_stack.shape[-2:], mask_stack_predicted.shape[-2:])

                x_min, y_min, x_max, y_max = self.cheetah_geom_list[idx_panel]

                x += x_min
                y += y_min

                peak_list.append((y, x))

            time_end = time.time()
            time_delta = time_end - time_start
            time_delta_name = 'pf:Convert coords'
            print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        return peak_list


    def count_hl_pixel_and_perf(self, img_stack, threshold_prob = 1 - 1e-4, min_num_peaks = 15):
        import time

        peak_list = []

        time_start = time.time()
        # Normalize the image stack...
        img_stack = (img_stack - img_stack.mean(axis = (2, 3), keepdim = True)) / img_stack.std(axis = (2, 3), keepdim = True)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Normalization'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Get activation feature map given the image stack...
        self.model.eval()
        with torch.no_grad():
            fmap_stack = self.model.forward(img_stack)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Inference'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Convert to probability with the sigmoid function...
        mask_stack_predicted = fmap_stack.sigmoid()
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Sigmoid'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Thresholding the probability...
        ## is_background = mask_stack_predicted < threshold_prob
        ## mask_stack_predicted[ is_background ] = 0
        ## mask_stack_predicted[~is_background ] = 1

        mask_stack_predicted = (mask_stack_predicted >= threshold_prob).type(torch.int32)

        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Thresholding'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        ## time_start = time.time()
        ## # Find center of mass for each image in the stack...
        ## num_stack, _, size_y, size_x = mask_stack_predicted.shape
        ## ## peak_pos_predicted_stack = self.calc_batch_center_of_mass(mask_stack_predicted.view(num_stack, size_y, size_x))
        ## peak_pos_predicted_stack = self.calc_batch_center_of_mass_perf(mask_stack_predicted.view(num_stack, size_y, size_x))
        ## ## peak_pos_predicted_stack = self.calc_batch_mean_position(mask_stack_predicted.view(num_stack, size_y, size_x))
        ## time_end = time.time()
        ## time_delta = time_end - time_start
        ## time_delta_name = 'pf:Center of mass'
        ## print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        ## # A workaround to avoid copying gpu memory to cpu when num of peaks is small...
        ## if len(peak_pos_predicted_stack) >= min_num_peaks:
        ##     time_start = time.time()
        ##     # Convert to cheetah coordinates...
        ##     for peak_pos in peak_pos_predicted_stack:
        ##         idx_panel, y, x = peak_pos.get()

        ##         if isnan(y) or isnan(x): continue

        ##         idx_panel = int(idx_panel)

        ##         y, x = coord_crop_to_img((y, x), img_stack.shape[-2:], mask_stack_predicted.shape[-2:])

        ##         x_min, y_min, x_max, y_max = self.cheetah_geom_list[idx_panel]

        ##         x += x_min
        ##         y += y_min

        ##         peak_list.append((y, x))

        ##     time_end = time.time()
        ##     time_delta = time_end - time_start
        ##     time_delta_name = 'pf:Convert coords'
        ##     print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        return []
