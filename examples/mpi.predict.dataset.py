#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
import h5py
import time
import pickle
import cupy as cp

from mpi4py import MPI
from cupyx.scipy import ndimage

from peaknet.predictor          import CheetahPeakFinder
from peaknet.datasets.utils     import PsanaImg
from peaknet.utils              import split_list_into_chunk
from peaknet.datasets.transform import center_crop, coord_crop_to_img
from peaknet.methods.unet       import UNet
from peaknet.model              import ConfigPeakFinderModel, PeakFinderModel


seed = 0

# Set up MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_data_tag = 11


# Set up experiments...
exp           = 'cxic0415'
run           = 101
img_load_mode = 'calib'
access_mode   = 'idx'
detector_name = 'CxiDs1.0:Cspad.0'
photon_energy = 12688.890590380644    # eV
encoder_value = -450.0034
psana_img = PsanaImg(exp, run, access_mode, detector_name)


# Load trained model...
timestamp = "2022_1101_2326_41"    # Manual
base_channels = 8
pos_weight    = 1.0
focal_alpha   = 0.8
focal_gamma   = 2.0
method = UNet( in_channels = 1, out_channels = 1, base_channels = base_channels )
config_peakfinder = ConfigPeakFinderModel( method = method, 
                                           pos_weight = pos_weight, 
                                           focal_alpha = focal_alpha,
                                           focal_gamma = focal_gamma)
model = PeakFinderModel(config_peakfinder)
model.init_params(from_timestamp = timestamp)   # Run this will load a trained model

# Load model to gpus if available...
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
model  = torch.nn.DataParallel(model.method).to(device)


# Load the peak finder...
path_cheetah_geom = 'cheetah_geom.pickle'
pf = CheetahPeakFinder(model = model, path_cheetah_geom = path_cheetah_geom)


# Finding (with MPI)...
# How many events to process?
num_events = len(psana_img.timestamps)
event_in_chunk = split_list_into_chunk(range(num_events), max_num_chunk = mpi_size)


# Define the MPI job for each worker
def find_peak(event_list, min_num_peaks):
    multipanel_mask     = psana_img.create_bad_pixel_mask()
    event_filtered_list = []
    for event in event_list:
        print(f"Processing {event:06d} : ", end = '')

        multipanel_img        = psana_img.get(event, None, 'calib')
        multipanel_img_masked = multipanel_mask * multipanel_img

        img_stack = torch.tensor(multipanel_img_masked).type(dtype=torch.float)[:,None].to(device)

        time_start = time.time()
        peak_list = pf.find_peak(img_stack, threshold_prob = 1 - 1e-4, min_num_peaks = min_num_peaks)
        time_end = time.time()
        time_delta = time_end - time_start
        print(f"Time delta: {time_delta * 1e3:.4f} millisecond.")

        if len(peak_list) < min_num_peaks: continue

        event_filtered_list.append([event, peak_list])

    return event_filtered_list

# Some peak finding criteria
min_num_peaks = 20

# MPI manager, but as worker too
if mpi_rank == 0:
    event_in_chunk_per_worker = event_in_chunk[mpi_rank]
    event_filtered_list = find_peak(event_in_chunk_per_worker, min_num_peaks = min_num_peaks)

    for i in range(1, mpi_size, 1):
        data_received = mpi_comm.recv(source = i, tag = mpi_data_tag)
        event_filtered_list.extend(data_received)

# MPI worker
if mpi_rank != 0:
    event_in_chunk_per_worker = event_in_chunk[mpi_rank]
    event_filtered_list = find_peak(event_in_chunk_per_worker, min_num_peaks = min_num_peaks)

    data_to_send = event_filtered_list
    mpi_comm.send(data_to_send, dest = 0, tag = mpi_data_tag)


if mpi_rank == 0:
    MPI.Finalize()

    drc_pf = 'demo'
    fl_event_filtered_list = f"pf.{timestamp}.peaks.pickle"
    path_event_filtered_list = os.path.join(drc_pf, fl_event_filtered_list)
    with open(path_event_filtered_list, 'wb') as fh:
        pickle.dump(event_filtered_list, fh, protocol = pickle.HIGHEST_PROTOCOL)

