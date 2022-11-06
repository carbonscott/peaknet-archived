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
## timestamp_prev = "2022_1026_1049_26"

# Set up parameters for an experiment...
drc_dataset   = 'datasets'
## fl_dataset    = 'sfx.0003.npy'
fl_dataset    = 'sfx.0000.raw.npy'    # Raw, just give it a try
path_dataset  = os.path.join(drc_dataset, fl_dataset)

frac_train    = 0.6
frac_validate = 0.5
dataset_usage = 'train'

base_channels = 8
pos_weight    = 1.0    # [IMPROVE] Remove it.
focal_alpha   = 1.2
focal_gamma   = 2.0

size_batch = 10
lr         = 5*1e-5
seed       = 0

# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            Fraction    (train)    : {frac_train}
            Fraction    (validate) : {frac_validate}
            Batch  size            : {size_batch}
            lr                     : {lr}
            pos_weight             : {pos_weight}
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
data_train   , data_val_and_test = split_dataset(dataset_list     , frac_train   , seed = seed)
data_validate, data_test         = split_dataset(data_val_and_test, frac_validate, seed = seed)

# Define the training set
dataset_train = SFXDataset( dataset_list = data_train,
                            seed         = seed, )

# Define validation set...
dataset_validate = SFXDataset( dataset_list = data_validate,
                               seed         = seed, )


# [[[ IMAGE ENCODER ]]]
# Config the encoder...
method = UNet( base_channels = base_channels, 
               in_channels   = 1, 
               out_channels  = 1, )


# [[[ MODEL ]]]
# Config the model...
config_peakfinder = ConfigPeakFinderModel( method     = method,
                                           pos_weight = pos_weight,
                                           focal_alpha = focal_alpha,
                                           focal_gamma = focal_gamma,
                                         )
model = PeakFinderModel(config_peakfinder)
model.init_params(from_timestamp = timestamp_prev)


# [[[ TRAINER ]]]
# Config the trainer...
config_train = ConfigTrainer( num_workers    = 0,
                              batch_size     = size_batch,
                              pin_memory     = True,
                              shuffle        = False,
                              lr             = lr, )

# Training...
trainer = Trainer(model, dataset_train, config_train)


# [[[ VALIDATOR ]]]
config_validator = ConfigValidator( num_workers    = 0,
                                    batch_size     = size_batch,
                                    pin_memory     = True,
                                    shuffle        = False,
                                    lr             = lr, )
validator = LossValidator(model, dataset_validate, config_validator)


# [[[ EPOCH MANAGER ]]]
max_epochs = 1000
epoch_manager = EpochManager( trainer   = trainer,
                              validator = validator,
                              timestamp = timestamp, )

epoch_manager.set_layer_to_capture(
    module_name_capture_list  = ["final_conv"],
    module_layer_capture_list = [torch.nn.ReLU],
)

freq_save = 5
for epoch in tqdm.tqdm(range(max_epochs)):
    epoch_manager.run_one_epoch(epoch = epoch)

    if epoch % freq_save == 0: 
        epoch_manager.save_model_parameters()
        epoch_manager.save_model_gradients()
        epoch_manager.save_state_dict()
