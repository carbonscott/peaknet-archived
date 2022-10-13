#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
import h5py

import time

from peaknet.datasets.psana_dataset import SFXDataset, ConfigDataset
from peaknet.methods.unet_simple    import UNet
from peaknet.model                  import ConfigPeakFinderModel, PeakFinderModel
from peaknet.predictor              import Predictor, ConfigPredictor

seed = 0

# [[[ DATASET ]]]
## timestamp = "2022_0908_1013_52"    # frac_train = 0.5, pos_weight = 2, pretty good.
timestamp = "2022_1012_1131_43"    # frac_train = 0.5, pos_weight = 2, pretty good.

base_channels = 8

size_batch = 21
## size_batch = 1

exp           = 'cxic0415'
run           = 101
img_load_mode = 'calib'

photon_energy = 12688.890590380644    # eV
encoder_value = -450.0034
adu_threshold = 100

access_mode   = 'idx'
detector_name = 'CxiDs1.0:Cspad.0'

# Config the dataset...
config_dataset = ConfigDataset( exp            = exp,
                                run            = run,
                                access_mode    = access_mode,
                                detector_name  = detector_name,
                                img_load_mode  = img_load_mode,
                                seed           = seed,
                                adu_threshold  = adu_threshold )

# Define the training set
dataset = SFXDataset(config_dataset)


# [[[ MODEL ]]]
# Config the model...
method = UNet(in_channels = 1, out_channels = 1, base_channels = base_channels)
pos_weight = 2.0
config_peakfinder = ConfigPeakFinderModel( method     = method,
                                           pos_weight = pos_weight, )
model = PeakFinderModel(config_peakfinder)

# Initialize weights...
def init_weights(module):
    if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
        module.weight.data.normal_(mean = 0.0, std = 0.02)
model.apply(init_weights)

# Define chkpt...
drc_cwd          = os.getcwd()
DRCCHKPT         = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
if not os.path.exists(prefixpath_chkpt): os.makedirs(prefixpath_chkpt)
path_chkpt = None
if timestamp is not None:
    fl_chkpt   = f"{timestamp}.train.chkpt"
    path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)


# [[[ PREDICTOR ]]]
config_predictor = ConfigPredictor( path_chkpt     = path_chkpt,
                                    num_workers    = 1,
                                    batch_size     = size_batch,
                                    pin_memory     = True,
                                    shuffle        = False, )
predictor = Predictor(model, dataset, config_predictor)
predictor.predict()
