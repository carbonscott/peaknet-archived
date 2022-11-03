#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import pickle
import numpy as np
import h5py
import time
import cupy as cp

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
        structure[1] = cp.array([[0,1,0],
                                 [1,1,1],
                                 [0,1,0]])

        # Fetch labels...
        batch_label, batch_num_feature = ndimage.label(batch_mask, structure = structure)

        # Calculate batch center of mass...
        batch_center_of_mass = ndimage.center_of_mass(batch_mask, batch_label, cp.asarray(range(1, batch_num_feature+1)))

        return batch_center_of_mass


    def find_peak(self, img_stack, threshold_prob = 1 - 1e-4):
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
        mask_stack_predicted[  mask_stack_predicted < threshold_prob ] = 0
        mask_stack_predicted[~(mask_stack_predicted < threshold_prob)] = 1

        # Find center of mass for each image in the stack...
        num_stack, _, size_y, size_x = mask_stack_predicted.shape
        peak_pos_predicted_stack = self.calc_batch_center_of_mass(mask_stack_predicted.view(num_stack, size_y, size_x))

        # Convert to cheetah coordinates...
        for idx_panel, y, x in peak_pos_predicted_stack:
            if cp.isnan(y) or cp.isnan(x): continue

            idx_panel = int(idx_panel)

            # For some reason, it's faster to do it on cpu
            x = x.get()
            y = y.get()

            y, x = coord_crop_to_img((y, x), img_stack.shape[-2:], mask_stack_predicted.shape[-2:])

            x_min, y_min, x_max, y_max = self.cheetah_geom_list[idx_panel]

            x += x_min
            y += y_min

            peak_list.append((y, x))

        return peak_list
