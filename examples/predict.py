#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
import h5py
import time
import pickle
import yaml
import argparse

from cupyx.scipy import ndimage
import cupy as cp

from peaknet.predictor          import CheetahPeakFinder
from peaknet.datasets.utils     import PsanaImg
from peaknet.datasets.transform import center_crop
from peaknet.methods.att_unet   import AttentionUNet
from peaknet.model              import ConfigPeakFinderModel, PeakFinderModel

# [[[ ARG PARSE ]]]
parser = argparse.ArgumentParser(description='Process a yaml file.')
parser.add_argument('yaml', help='The input yaml file.')
args = parser.parse_args()

# [[[ Configure ]]]
fl_yaml = args.yaml
basename_yaml = fl_yaml[:fl_yaml.rfind('.yaml')]

# Load the YAML file
with open(fl_yaml, 'r') as fh:
    config = yaml.safe_load(fh)

# Access the values
# ___/ PeakNet model \___
timestamp            = config['timestamp']
epoch                = config['epoch'    ]
tag                  = config['tag'      ]
uses_skip_connection = config['uses_skip_connection']

# ___/ Experimental data \___
# Psana...
exp           = config['exp'          ]
run           = config['run'          ]
img_load_mode = config['img_load_mode']
access_mode   = config['access_mode'  ]
detector_name = config['detector_name']
photon_energy = config['photon_energy']
encoder_value = config['encoder_value']

# Data range...
event_min     = config['event_min']
event_max     = config['event_max']

# ___/ Output \___
dir_results = config["dir_results"]

# ___/ Misc \___
path_cheetah_geom = config["path_cheetah_geom"]

# Set up experiments...
psana_img = PsanaImg(exp, run, access_mode, detector_name)

# Load trained model...
fl_chkpt = None if timestamp is None else f"{timestamp}.epoch_{epoch}{tag}.chkpt"

base_channels = 8
focal_alpha   = 1.2
focal_gamma   = 2.0
method = AttentionUNet( base_channels        = base_channels,
                        in_channels          = 1,
                        out_channels         = 3,
                        uses_skip_connection = uses_skip_connection,
                        att_gate_channels    = None, )
config_peakfinder = ConfigPeakFinderModel( method = method, 
                                           focal_alpha = focal_alpha,
                                           focal_gamma = focal_gamma)
model = PeakFinderModel(config_peakfinder)
model.init_params(fl_chkpt = fl_chkpt)   # Run this will load a trained model

# Load model to gpus if available...
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
model  = torch.nn.DataParallel(model.method).to(device)


# Load the peak finder...
pf = CheetahPeakFinder(model = model, path_cheetah_geom = path_cheetah_geom)


# Finding...
min_num_peaks = 15
if event_min is None: event_min = 0
if event_max is None: event_max = len(psana_img.timestamps)
## multipanel_mask     = psana_img.create_bad_pixel_mask()
event_filtered_list = []
is_init = True
for enum_idx, event in enumerate(range(event_min, event_max)):
    print(f"___/ Event {event:06d} \___")

    img = psana_img.get(event, None, img_load_mode)

    if img is None: continue

    # [DATA SPECIFIC] This Rayonix has sharp edges
    if is_init:
        mask = np.zeros_like(img)

        offset = 10
        size_y, size_x = img.shape[-2:]
        xmin = 0 + offset
        xmax = size_x - offset
        ymin = 0 + offset
        ymax = size_y - offset

        mask[ymin:ymax, xmin:xmax] = 1.0

        is_init = False

    img *= mask

    img = torch.tensor(img).type(dtype=torch.float)[None,None].to(device)

    # Normalization is done in find_peak below

    time_start = time.time()
    peak_list = pf.find_peak_w_softmax(img, min_num_peaks = min_num_peaks)
    time_end = time.time()
    time_delta = time_end - time_start
    print(f"Time delta: {time_delta * 1e3:.4f} millisecond.")

    if len(peak_list) < min_num_peaks: continue

    event_filtered_list.append([event, peak_list])


fl_event_filtered_list = f"{basename_yaml}.peaks"
path_event_filtered_list = os.path.join(dir_results, fl_event_filtered_list)
with open(path_event_filtered_list, 'wb') as fh:
    pickle.dump(event_filtered_list, fh, protocol = pickle.HIGHEST_PROTOCOL)
