#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
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
fl_dataset    = 'cxilz0720.0001.npy'
path_dataset  = os.path.join(drc_dataset, fl_dataset)

lr_exponents = np.linspace(-7, -1, 100)

frac_train    = 0.7
## frac_validate = 0.5
dataset_usage = 'train'

base_channels = 8
focal_alpha   = 1.2
focal_gamma   = 2.0

size_batch = 100
lr         = 5*1e-5
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


# Scan learning rate...
loss_by_lr_list = []
for enum_idx, lr_exponent in enumerate(tqdm.tqdm(lr_exponents, disable=False)):
    lr = 10 ** lr_exponent
    trainer.config.lr = lr
    validator.config.lr = lr

    ## # [[[ TRAIN EPOCHS ]]]
    ## loss_train_hist    = []
    ## loss_validate_hist = []
    ## loss_min_hist      = []

    # [[[ EPOCH MANAGER ]]]
    epoch_manager = EpochManager( trainer   = trainer,
                                  validator = validator, )

    max_epochs = 1
    for epoch in range(max_epochs):
        loss_train, loss_validate, loss_min = epoch_manager.run_one_epoch(epoch, returns_loss = True)

        ## loss_train_hist.append(loss_train)
        ## loss_validate_hist.append(loss_validate)
        ## loss_min_hist.append(loss_min)

    loss_by_lr_list.append([float(lr_exponent), float(loss_train), float(loss_validate), float(loss_min)])

    if enum_idx % 50 == 0:
        fl_pickle = "find_lr.pickle"
        with open(fl_pickle, 'wb') as fh:
            pickle.dump(loss_by_lr_list, fh)

fl_pickle = "find_lr.pickle"
with open(fl_pickle, 'wb') as fh:
    pickle.dump(loss_by_lr_list, fh)
