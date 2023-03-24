#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import torch
import socket
import tqdm
import numpy as np
from peaknet.datasets.lite  import SFXDataset
from peaknet.methods.unet   import UNet
from peaknet.model          import ConfigPeakFinderModel, PeakFinderModel
from peaknet.trainer        import ConfigTrainer, Trainer
from peaknet.validator      import ConfigValidator, LossValidator
from peaknet.utils          import init_logger, EpochManager, MetaLog, split_dataset

timestamp_prev = None
epoch = None
fl_chkpt = None if timestamp_prev is None else f"{timestamp_prev}.epoch_{epoch}.chkpt"


# Set up parameters for an experiment...
drc_dataset   = 'datasets'
fl_dataset    = 'mfxp22820.0001.npy'
path_dataset  = os.path.join(drc_dataset, fl_dataset)

frac_train    = 0.6
## frac_validate = 0.5
dataset_usage = 'train'

base_channels = 8
focal_alpha   = 1.2
focal_gamma   = 2.0

size_batch = 3
lr         = 10**(-4.25)
seed       = 0

# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            Fraction    (train)    : {frac_train}
            Batch  size            : {size_batch}
            lr                     : {lr}
            base_channels          : {base_channels}
            focal_alpha            : {focal_alpha}
            focal_gamma            : {focal_gamma}
            continued training???  : from {timestamp_prev}

            """

# [[[ LOGGING ]]]
timestamp = init_logger(returns_timestamp = True)

# Create a metalog to the log file, explaining the purpose of this run...
metalog = MetaLog( comments = comments )
metalog.report()


# [[[ DATASET ]]]
# Load raw data...
dataset_list = np.load(path_dataset)
data_train, data_validate = split_dataset(dataset_list, frac_train, seed = seed)
## data_train   , data_val_and_test = split_dataset(dataset_list     , frac_train   , seed = seed)
## data_validate, data_test         = split_dataset(data_val_and_test, frac_validate, seed = seed)

# Define the training set
dataset_train = SFXDataset(dataset_list = data_train)

# Define validation set...
dataset_validate = SFXDataset(dataset_list = data_validate)


# [[[ IMAGE ENCODER ]]]
# Config the encoder...
method = UNet( base_channels = base_channels, 
               in_channels   = 1, 
               out_channels  = 1, )


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
                              num_workers  = 0,
                              batch_size   = size_batch,
                              pin_memory   = True,
                              shuffle      = False,
                              tqdm_disable = True,
                              lr           = lr, )

# Training...
trainer = Trainer(model, dataset_train, config_train)


# [[[ VALIDATOR ]]]
config_validator = ConfigValidator( num_workers  = 0,
                                    batch_size   = size_batch,
                                    pin_memory   = True,
                                    shuffle      = False,
                                    tqdm_disable = True,
                                    lr           = lr, )
validator = LossValidator(model, dataset_validate, config_validator)


# [[[ EPOCH MANAGER ]]]
max_epochs = 3000
epoch_manager = EpochManager( trainer   = trainer,
                              validator = validator, )

## epoch_manager.set_layer_to_capture(
##     module_name_capture_list  = ["final_conv"],
##     module_layer_capture_list = [torch.nn.ReLU],
## )

print(timestamp)
freq_save = 5
for epoch in tqdm.tqdm(range(max_epochs)):
    loss_train, loss_validate, loss_min = epoch_manager.run_one_epoch(epoch, returns_loss = True)

    ## if epoch % freq_save == 0: 
    ##     epoch_manager.save_model_parameters()
    ##     epoch_manager.save_model_gradients()
    ##     epoch_manager.save_state_dict()
