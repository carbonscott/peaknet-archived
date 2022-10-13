#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import random
import os
import logging
import numpy as np

## import time

from torch.utils.data import Dataset

from peaknet.utils          import set_seed, split_dataset
from peaknet.datasets.utils import PsanaImg

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
    SFX dataset consists of all events given a tuple (exp, run).  

    get_     method returns data by sequential index.
    extract_ method returns object by files.
    """

    def __init__(self, config):
        self.exp                   = getattr(config, 'exp'                  , None)
        self.run                   = getattr(config, 'run'                  , None)
        self.access_mode           = getattr(config, 'access_mode'          , None)
        self.detector_name         = getattr(config, 'detector_name'        , None)
        self.img_load_mode         = getattr(config, 'img_load_mode'        , None)    # calib or raw
        self.seed                  = getattr(config, 'seed'                 , None)
        self.add_channel_ok        = getattr(config, 'add_channel_ok'       , True)
        self.adu_threshold         = getattr(config, 'adu_threshold'        , 1000)
        self.min_num_peak_required = getattr(config, 'min_num_peak_required', 15)

        self.psana_img = PsanaImg(self.exp, self.run, self.access_mode, self.detector_name)

        self.mask = self.psana_img.create_bad_pixel_mask()


    def __len__(self):
        return len(self.psana_img)


    def __getitem__(self, idx):
        ## time_start = time.time()
        panel_list = self.psana_img.get(idx, None, self.img_load_mode)
        panel_list *= self.mask
        ## time_end = time.time()

        ## print(f'Time delta: {time_end - time_start} sec.')

        return panel_list
