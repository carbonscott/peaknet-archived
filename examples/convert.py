#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py
import pickle
import yaml
import argparse

from peaknet.datasets.utils import PsanaImg


def convert_psana_to_cheetah(panel_list):
    # [!!!] Hard code
    dim0 = 8 * 185
    dim1 = 4 * 388

    # Convert calib image to cheetah image
    img = np.zeros((dim0, dim1))
    counter = 0
    for quad in range(4):
        for seg in range(8):
            img[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = panel_list[counter, :, :]
            counter += 1

    return img


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

# ___/ Experimental data \___
# Psana...
exp           = config['exp'          ]
run           = config['run'          ]
img_load_mode = config['img_load_mode']
access_mode   = config['access_mode'  ]
detector_name = config['detector_name']
photon_energy = config['photon_energy']
encoder_value = config['encoder_value']

# ___/ Output \___
dir_results = config["dir_results"]


# [[[ MAIN ]]]
psana_img = PsanaImg(exp, run, access_mode, detector_name)

# Read peak from a peak file...
basename = f"{basename_yaml}.peaks"
path_peak_file = os.path.join(dir_results, f'{basename}')
with open(path_peak_file, 'rb') as handle:
    event_filtered_list = pickle.load(handle)

# Create cxi for indexing...
fl_cxi          = os.path.join(dir_results, f"{basename}.cxi")
max_num_peak    = 2048
num_event       = len(event_filtered_list)
bad_pixel_mask = psana_img.create_bad_pixel_mask()
with h5py.File(fl_cxi, 'w') as myHdf5:
    # [!!!] Hard code
    dim0 = 1920
    dim1 = 1920

    grpName     = "/entry_1/result_1"
    dset_nPeaks = "/nPeaks"
    dset_posX   = "/peakXPosRaw"
    dset_posY   = "/peakYPosRaw"
    dset_atot   = "/peakTotalIntensity"

    grp = myHdf5.create_group(grpName)
    myHdf5.create_dataset(grpName + dset_nPeaks, (num_event,             ), dtype='int')
    myHdf5.create_dataset(grpName + dset_posX  , (num_event, max_num_peak), dtype='float32', chunks=(1, max_num_peak))
    myHdf5.create_dataset(grpName + dset_posY  , (num_event, max_num_peak), dtype='float32', chunks=(1, max_num_peak))
    myHdf5.create_dataset(grpName + dset_atot  , (num_event, max_num_peak), dtype='float32', chunks=(1, max_num_peak))

    myHdf5.create_dataset("/LCLS/detector_1/EncoderValue", (1,), dtype=float)
    myHdf5.create_dataset("/LCLS/photon_energy_eV", (1,), dtype=float)
    dset = myHdf5.create_dataset("/entry_1/data_1/data", (num_event, dim0, dim1), dtype=np.float32) # change to float32
    ## dsetM = myHdf5.create_dataset("/entry_1/data_1/mask", (dim0, dim1), dtype='int')

    for seqi, (event, peak_per_event_list) in enumerate(event_filtered_list):
        # Save this event???
        nPeaks = len(peak_per_event_list)
        if nPeaks > max_num_peak: continue

        # Save images...
        img = psana_img.get(event, None, 'calib')
        img_masked = bad_pixel_mask * img
        dset[seqi] = img_masked

        # Save peaks...
        for i, peak in enumerate(peak_per_event_list):
            cheetahRow, cheetahCol = peak
            myHdf5[grpName + dset_posX][seqi, i] = cheetahCol
            myHdf5[grpName + dset_posY][seqi, i] = cheetahRow
        myHdf5[grpName + dset_nPeaks][seqi] = nPeaks

    myHdf5["/LCLS/detector_1/EncoderValue"][0] = encoder_value  # mm
    myHdf5["/LCLS/photon_energy_eV"][0] = photon_energy



fl_lst = os.path.join(dir_results, f"{basename}.lst")
with open(fl_lst,'w') as fh:
    for i, (event, peak_per_event_list) in enumerate(event_filtered_list):
        # Index this event???
        nPeaks = len(peak_per_event_list)
        if nPeaks > max_num_peak: continue

        fh.write(f"{fl_cxi} //{i}")
        fh.write("\n")
