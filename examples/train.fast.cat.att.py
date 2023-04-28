#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import torch
import socket
import tqdm
import numpy as np
from peaknet.datasets.lite    import SFXMulticlassDataset
from peaknet.methods.att_unet import AttentionUNet
from peaknet.model            import ConfigPeakFinderModel, PeakFinderModel
from peaknet.trainer          import ConfigTrainer, Trainer
from peaknet.validator        import ConfigValidator, LossValidator
from peaknet.utils            import init_logger, EpochManager, MetaLog, split_dataset

from peaknet.aug import RandomShift,  \
                        RandomRotate, \
                        RandomPatch

torch.autograd.set_detect_anomaly(False)

## timestamp_prev = "2023_0425_2349_15"
## epoch = 161
## timestamp_prev = "2023_0426_2111_50"
## epoch = 7
## timestamp_prev = "2023_0426_2225_04"
## epoch = 250
timestamp_prev = None
epoch = None
fl_chkpt = None if timestamp_prev is None else f"{timestamp_prev}.epoch_{epoch}.chkpt"


# Set up parameters for an experiment...
drc_dataset  = 'datasets'
## fl_dataset   = 'mfx13016.0028.npy'
## fl_dataset   = 'mfx13016_0028+mfxp22820_0013.data.npy'
## fl_dataset   = 'mfx13016_0028_N68+mfxp22820_0013_N37.data.npy'
## fl_dataset   = 'mfx13016_0028_N68+mfxp22820_0013_N37+mfx13016_0028_N63_low_photon.data.npy'
## fl_dataset   = 'mfx13016_0028_N68+mfxp22820_0013_N37+mfx13016_0028_N63_low_photon.68v20v20.data.npy'    # size_sample = 2400
fl_dataset   = 'mfx13016_0028_N68+mfxp22820_0013_N37+mfx13016_0028_N63_low_photon.68v37v30.data.npy'       # size_sample = 3000
path_dataset = os.path.join(drc_dataset, fl_dataset)

size_sample   = 3000
## size_sample   = 20    # Quick check if the program runs
frac_train    = 0.8
frac_validate = 1.0
dataset_usage = 'train'

uses_skip_connection = True    # Default: True

base_channels = 8
focal_alpha   = 1.2 * 10**(0)
focal_gamma   = 2 * 10**(0)

size_batch  = 10 * 3
num_workers = 12
lr          = 10**(-4.25)
seed        = 0

# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            File (Dataset)         : {path_dataset}
            Fraction    (train)    : {frac_train}
            Dataset size           : {size_sample}
            Batch  size            : {size_batch}
            lr                     : {lr}
            base_channels          : {base_channels}
            focal_alpha            : {focal_alpha}
            focal_gamma            : {focal_gamma}
            uses_skip_connection   : {uses_skip_connection}
            continued training???  : from {fl_chkpt}

            """

# [[[ LOGGING ]]]
timestamp = init_logger(returns_timestamp = True)

# Create a metalog to the log file, explaining the purpose of this run...
metalog = MetaLog( comments = comments )
metalog.report()


# [[[ DATASET ]]]
# Load raw data...
dataset_list = np.load(path_dataset, allow_pickle = True)
## data_train, data_validate = split_dataset(dataset_list, frac_train, seed = seed)
data_train   , data_val_and_test = split_dataset(dataset_list     , frac_train   , seed = seed)
data_validate, data_test         = split_dataset(data_val_and_test, frac_validate, seed = seed)

# Set up transformation rules
num_patch      = 50
size_patch     = 100
frac_shift_max = 0.2
angle_max      = 360

trans_list = (
    RandomRotate(angle_max = angle_max, order = 0),
    RandomShift(frac_shift_max, frac_shift_max),
    RandomPatch(num_patch = num_patch, size_patch_y = size_patch, size_patch_x = size_patch, var_patch_y = 0.2, var_patch_x = 0.2),
)

# Define the training set
dataset_train = SFXMulticlassDataset(data_list          = data_train,
                                     size_sample        = size_sample,
                                     trans_list         = trans_list,
                                     normalizes_data    = True,
                                     prints_cache_state = True,
                                     mpi_comm           = None, )

# Define validation set...
dataset_validate = SFXMulticlassDataset(data_list          = data_validate,
                                        size_sample        = size_sample//2,
                                        trans_list         = trans_list,
                                        normalizes_data    = True,
                                        prints_cache_state = True,
                                        mpi_comm           = None, )


# [[[ IMAGE ENCODER ]]]
# Config the encoder...
method = AttentionUNet( base_channels        = base_channels,
                        in_channels          = 1,
                        out_channels         = 3,
                        uses_skip_connection = uses_skip_connection,
                        att_gate_channels    = None, )


# [[[ MODEL ]]]
# Config the model...
config_peakfinder = ConfigPeakFinderModel( method      = method,
                                           focal_alpha = focal_alpha,
                                           focal_gamma = focal_gamma,
                                         )
model = PeakFinderModel(config_peakfinder)
model.init_params(fl_chkpt = fl_chkpt)


# [[[ TRAINER ]]]
# Config the trainer...
config_train = ConfigTrainer( timestamp    = timestamp, 
                              num_workers  = num_workers,
                              batch_size   = size_batch,
                              pin_memory   = True,
                              shuffle      = False,
                              tqdm_disable = False,
                              lr           = lr,
                              uses_mixed_precision = True, )

# Training...
trainer = Trainer(model, dataset_train, config_train)


# [[[ VALIDATOR ]]]
config_validator = ConfigValidator( num_workers  = num_workers,
                                    batch_size   = size_batch,
                                    pin_memory   = True,
                                    shuffle      = False,
                                    tqdm_disable = False,
                                    lr           = lr,
                                    uses_mixed_precision = True, )
validator = LossValidator(model, dataset_validate, config_validator)


# [[[ EPOCH MANAGER ]]]
max_epochs = 3000
epoch_manager = EpochManager( trainer   = trainer,
                              validator = validator, )

## epoch_manager.set_layer_to_capture(
##     module_name_capture_list  = ["final_conv"],
##     module_layer_capture_list = [torch.nn.ReLU],
## )

print(timestamp, flush = True)
freq_save = 5
for epoch in tqdm.tqdm(range(max_epochs)):
    loss_train, loss_validate, loss_min = epoch_manager.run_one_epoch(epoch, returns_loss = True)

    ## if epoch % freq_save == 0: 
    ##     epoch_manager.save_model_parameters()
    ##     epoch_manager.save_model_gradients()
    ##     epoch_manager.save_state_dict()
