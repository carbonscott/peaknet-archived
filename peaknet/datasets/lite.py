#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import os
import logging

from torch.utils.data import Dataset

from peaknet.utils                  import set_seed, split_dataset

logger = logging.getLogger(__name__)

class ConfigDataset:
    ''' Biolerplate code to config dataset classs'''

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items(): setattr(self, k, v)


    def report(self):
        logger.info(f"___/ Configure Dataset \___")
        for k, v in self.__dict__.items():
            if k == 'kwargs': continue
            logger.info(f"KV - {k:16s} : {v}")




class SFXDataset(Dataset):
    """
    SFX images are collected from multiple datasets specified in the input csv
    file. All images are organized in a plain list.  

    get_     method returns data by sequential index.
    extract_ method returns object by files.
    """

    def __init__(self, dataset_list, trans = None, seed = None):
        self.dataset_list = dataset_list
        self.trans        = trans
        self.seed         = seed

        return None


    def __len__(self):
        return len(self.dataset_list)


    def __getitem__(self, idx):
        img, label = self.dataset_list[idx]

        # Apply any possible transformation...
        # How to define a custom transform function?
        # Input : img, **kwargs 
        # Output: img_transfromed
        if self.trans is not None:
            img = self.trans(img)

        # Normalize input image...
        img_mean = np.mean(img)
        img_std  = np.std(img)
        img      = (img - img_mean) / img_std

        return img, label
