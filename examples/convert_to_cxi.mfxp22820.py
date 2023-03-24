#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py
import pickle

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


timestamp = "2023_0321_1052_03"
event_min = 0
event_max = 9000

exp           = 'mfxp22820'
## run           = 13
run           = 14
img_load_mode = 'calib'
access_mode   = 'idx'
detector_name = 'Rayonix'
photon_energy = 9.54e3    # eV
encoder_value = 300       # clen

psana_img = PsanaImg(exp, run, access_mode, detector_name)


basename = f'pf.{timestamp}.{exp}.run{run}.{event_min:06d}-{event_max:06d}.peaks'
path_peak_file = f'demo/{basename}.pickle'
with open(path_peak_file, 'rb') as handle:
    event_filtered_list = pickle.load(handle)



fl_cxi          = f'demo/{basename}.cxi'
max_num_peak    = 2048
num_event       = len(event_filtered_list)
## multipanel_mask = psana_img.create_bad_pixel_mask()
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
        ## img_masked = multipanel_mask * img
        dset[seqi] = img

        # Save peaks...
        for i, peak in enumerate(peak_per_event_list):
            cheetahRow, cheetahCol = peak
            myHdf5[grpName + dset_posX][seqi, i] = cheetahCol
            myHdf5[grpName + dset_posY][seqi, i] = cheetahRow
        myHdf5[grpName + dset_nPeaks][seqi] = nPeaks

    myHdf5["/LCLS/detector_1/EncoderValue"][0] = encoder_value  # mm
    myHdf5["/LCLS/photon_energy_eV"][0] = photon_energy



fl_lst = f'demo/{basename}.lst'
with open(fl_lst,'w') as fh:
    for i, (event, peak_per_event_list) in enumerate(event_filtered_list):
        # Index this event???
        nPeaks = len(peak_per_event_list)
        if nPeaks > max_num_peak: continue

        fh.write(f"{fl_cxi} //{i}")
        fh.write("\n")
