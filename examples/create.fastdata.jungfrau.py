#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os
import sys
import torch
from peaknet.datasets.fastdata  import ConfigDataset, SFXPanelDatasetMini

# Set up parameters for an experiment...
basename             =  "cxilz0720"
fl_csv               = f"{basename}.csv"
drc_project          = os.getcwd()
size_sample_train    = 1000
size_sample_validate = 1000
frac_train           = 1.0
frac_validate        = None
dataset_usage        = 'train'

seed = 0

# [[[ DATASET ]]]
# Config the dataset...
config_dataset = ConfigDataset( fl_csv        = fl_csv,
                                drc_project   = drc_project,
                                size_sample   = size_sample_train,
                                dataset_usage = dataset_usage,
                                trans         = None,
                                frac_train    = frac_train,
                                frac_validate = frac_validate,
                                mpi_comm      = None,
                                seed          = seed,
                                mask_radius   = 3,
                                is_batch_mask = True,
                                snr_threshold = 1.0, )

# Define the training set
dataset_train = SFXPanelDatasetMini(config_dataset)

data_dump_list = []
for i, (img, mask) in enumerate(dataset_train):
    ## if i > 10: break

    print( f"Processing img {i:04d}..." )
    data_dump_list.append((img, mask))

drc_fastdata        = "fastdata"
drc_cwd             = os.getcwd()
prefixpath_fastdata = os.path.join(drc_cwd, drc_fastdata)
if not os.path.exists(prefixpath_fastdata): os.makedirs(prefixpath_fastdata)
fl_fastdata         = f"{basename}.human_engineered.fastdata"
path_fastdata       = os.path.join(prefixpath_fastdata, fl_fastdata)


with open(path_fastdata, 'wb') as fh:
    pickle.dump(data_dump_list, fh, protocol=pickle.HIGHEST_PROTOCOL)
