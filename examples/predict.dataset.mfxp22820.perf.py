#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
import h5py
import time
import pickle

from cupyx.scipy import ndimage
import cupy as cp

from peaknet.predictor          import CheetahPeakFinder
from peaknet.datasets.utils     import PsanaImg
from peaknet.datasets.transform import center_crop, coord_crop_to_img
from peaknet.methods.unet       import UNet
from peaknet.model              import ConfigPeakFinderModel, PeakFinderModel


# Set up experiments...
exp           = 'mfxp22820'
## run           = 13
run           = 14
img_load_mode = 'calib'
access_mode   = 'idx'
detector_name = 'Rayonix'
psana_img     = PsanaImg(exp, run, access_mode, detector_name)

threshold_prob = 0.5


# Load trained model...
timestamp = "2023_0321_1052_03"
epoch = 93
fl_chkpt = None if timestamp is None else f"{timestamp}.epoch_{epoch}.chkpt"

base_channels = 8
focal_alpha   = 1.2
focal_gamma   = 2.0
method = UNet( in_channels = 1, out_channels = 1, base_channels = base_channels )
config_peakfinder = ConfigPeakFinderModel( method = method, 
                                           focal_alpha = focal_alpha,
                                           focal_gamma = focal_gamma)
model = PeakFinderModel(config_peakfinder)
model.init_params(fl_chkpt = fl_chkpt)   # Run this will load a trained model

# Load model to gpus if available...
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
model  = torch.nn.DataParallel(model.method).to(device)


# Load the peak finder...
path_cheetah_geom = 'cheetah_geom.pickle'
pf = CheetahPeakFinder(model = model, path_cheetah_geom = path_cheetah_geom)


# Finding...
min_num_peaks       = 15
event_start         = 0
event_end           = len(psana_img.timestamps)
## event_end           = 500
## multipanel_mask     = psana_img.create_bad_pixel_mask()
event_filtered_list = []

acc_counter = 0
batch_size = 11
for enum_idx, event in enumerate(range(event_start, event_end)):
    print(f"___/ Event {event:06d} \___")

    img = psana_img.get(event, None, img_load_mode)

    # [DATA SPECIFIC] This Rayonix has sharp edges
    if enum_idx == 0: 
        mask = np.zeros_like(img)

        offset = 10
        size_y, size_x = img.shape
        xmin = 0 + offset
        xmax = size_x - offset
        ymin = 0 + offset
        ymax = size_y - offset

        mask[ymin:ymax, xmin:xmax] = 1.0

    img *= mask
    img = torch.tensor(img).type(dtype=torch.float)[None].to(device)

    # Prepare for data accumulation...
    if acc_counter == 0:
        img_stack = torch.zeros((batch_size, *img.shape), device = device)

    ## import pdb; pdb.set_trace()

    # Accumulate data...
    img_stack[acc_counter] = img

    acc_counter += 1

    # Time to predict...
    if acc_counter == batch_size:
        time_start = time.time()
        peak_list = pf.find_peak_and_perf(img_stack, threshold_prob = threshold_prob, min_num_peaks = min_num_peaks)
        time_end = time.time()
        time_delta = time_end - time_start
        print(f"Time delta: {time_delta * 1e3:.4f} millisecond.")

        acc_counter = 0

        if len(peak_list) < min_num_peaks: continue

        event_filtered_list.append([event, peak_list])



## drc_pf = 'demo'
## fl_event_filtered_list = f"pf.{timestamp}.{exp}.run{run}.{event_start:06d}-{event_end:06d}.peaks.pickle"
## path_event_filtered_list = os.path.join(drc_pf, fl_event_filtered_list)
## with open(path_event_filtered_list, 'wb') as fh:
##     pickle.dump(event_filtered_list, fh, protocol = pickle.HIGHEST_PROTOCOL)
