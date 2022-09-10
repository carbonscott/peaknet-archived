#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import random
import csv
import os
import logging

from torch.utils.data import Dataset

from peaknet.utils                  import set_seed
from peaknet.datasets.utils         import PsanaImg

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




class DatasetParser(Dataset):
    """
    SFX images are collected from multiple datasets specified in the input csv
    file. All images are organized in a plain list.  
    """

    def __init__(self, config):
        self.fl_csv         = getattr(config, 'fl_csv'        , None)
        self.seed           = getattr(config, 'seed'          , None)
        self.trans          = getattr(config, 'trans'         , None)
        self.mpi_comm       = getattr(config, 'mpi_comm'      , None)
        self.add_channel_ok = getattr(config, 'add_channel_ok', True)
        self.mask_radius    = getattr(config, 'mask_radius'   , 3   )
        self.snr_threshold  = getattr(config, 'snr_threshold' , 0.3 )
        self.adu_threshold  = getattr(config, 'adu_threshold' , 1000)

        # Variables that capture information in data spliting
        self.metadata_list = []

        # Variables for caching...
        self.psana_img_dict       = {} # psana img handler indexed by (exp, run)

        # Set the seed...
        # Debatable whether seed should be set in the dataset or in the running code
        if not self.seed is None: set_seed(self.seed)

        # Set up mpi...
        if self.mpi_comm is not None:
            self.mpi_size     = self.mpi_comm.Get_size()    # num of processors
            self.mpi_rank     = self.mpi_comm.Get_rank()
            self.mpi_data_tag = 11

        # Read run info from the csv...
        with open(self.fl_csv, 'r') as fh:
            lines = csv.reader(fh)

            next(lines)

            # Read each line/dataset...
            for line in lines:
                # Fetch metadata of a dataset 
                exp, run, access_mode, detector_name = line

                # Form a minimal basename to describe a dataset...
                basename = (exp, run)

                # Initiate image accessing layer...
                psana_img = PsanaImg(exp, run, access_mode, detector_name)
                self.psana_img_dict[basename] = psana_img

        return None




class SFXRandomSubset(DatasetParser):

    def __init__(self, config):
        super().__init__(config)

        self.size_sample    = getattr(config, 'size_sample', 1)
        self.add_channel_ok = getattr(config, 'add_channel_ok', False)

        self.PSANA_MODE  = 'calib'

        self.raw_img_cache_dict = {}
        self.img_cache_dict     = {}

        self.metadata_list = self.form_random_subset()


    def form_random_subset(self):
        # Select a finite number of events by choosing where (basename) they are from...
        basename_unique_list = list(self.psana_img_dict.keys())
        basename_sample_list = random.choices(basename_unique_list, k = self.size_sample) \
                               if   self.size_sample is not None                          \
                               else []

        # Fetch the total number of events per basename...
        num_event_per_basename_dict = { k : len(v) for k, v in self.psana_img_dict.items() }

        # Save a random event for each basename in the sample list as metadata...
        metadata_list = []
        for i, basename_sample in enumerate(basename_sample_list):
            # Fetch the total number of events for this basename...
            num_event = num_event_per_basename_dict[basename_sample]

            # Choose one event randomly within the range...
            event = random.randrange(num_event)

            # Estimate the number of panels...
            if i == 0:
                img = self.psana_img_dict[basename_sample].get(event, id_panel = None, mode = self.PSANA_MODE)
                num_panels = img.shape[0]

            # Choose one panel randomly within the range...
            panel = random.randrange(num_panels)

            # Save event...
            event_tuple = (*basename_sample, event, panel)
            metadata_list.append(event_tuple)

        return metadata_list


    def __len__(self):
        return len(self.metadata_list)


    def get_img(self, idx, verbose = False):
        # Read image...
        exp, run, event, panel = self.metadata_list[idx]
        basename = (exp, run)
        img = self.psana_img_dict[basename].get(int(event), id_panel = panel, mode = self.PSANA_MODE)

        if self.add_channel_ok: img = img[None,]

        if verbose: logger.info(f'DATA LOADING - {exp} {run} {event} {panel}.')

        return img


    def get_raw_img(self, idx, verbose = False):
        # Define local psana_mode
        psana_mode = "image"

        # Read image...
        exp, run, event, panel = self.metadata_list[idx]
        basename = (exp, run)

        # Cache raw images as they might reoccur often...
        if idx in self.raw_img_cache_dict:
            img = self.raw_img_cache_dict[idx]
        else:
            img = self.psana_img_dict[basename].get(int(event), id_panel = None, mode = psana_mode)
            self.raw_img_cache_dict[idx] = img

        if self.add_channel_ok: img = img[None,]

        if verbose: logger.info(f'DATA LOADING - {exp} {run} {event}.')

        return img


    def __getitem__(self, idx):
        img = self.img_cache_dict[idx]      \
              if idx in self.img_cache_dict \
              else self.get_img(idx)


        if self.trans is not None:
            img = self.trans(img)

        # Normalize input image...
        img_mean = np.mean(img)
        img_std  = np.std(img)
        img_norm = (img - img_mean) / img_std

        return img


    def cache_img(self):
        ''' Cache image.
        '''
        # Construct index list...
        idx_list = range(len(self.metadata_list))

        for idx in idx_list:
            # Skip those have been recorded...
            if idx in self.img_cache_dict: continue

            # Otherwise, record it...
            img = self.get_img(idx, verbose = True)
            self.img_cache_dict[idx] = img

        return None


    def mpi_cache_img(self):
        ''' Cache image in the seq_random_list unless a subset is specified
            using MPI.
        '''
        # Import chunking method...
        from peaknet.utils import split_list_into_chunk

        # Get the MPI metadata...
        mpi_comm     = self.mpi_comm
        mpi_size     = self.mpi_size
        mpi_rank     = self.mpi_rank
        mpi_data_tag = self.mpi_data_tag

        # If subset is not give, then go through the whole set...
        idx_list = range(len(self.metadata_list))

        # Split the workload...
        idx_list_in_chunk = split_list_into_chunk(idx_list, max_num_chunk = mpi_size)

        # Process chunk by each worker...
        # No need to sync the peak_cache_dict across workers
        if mpi_rank != 0:
            if mpi_rank < len(idx_list_in_chunk):
                idx_list_per_worker = idx_list_in_chunk[mpi_rank]
                img_cache_dict = self._mpi_cache_img_per_rank(idx_list_per_worker)

            mpi_comm.send(img_cache_dict, dest = 0, tag = mpi_data_tag)

        if mpi_rank == 0:
            idx_list_per_worker = idx_list_in_chunk[mpi_rank]
            self.img_cache_dict = self._mpi_cache_img_per_rank(idx_list_per_worker)

            for i in range(1, mpi_size, 1):
                img_cache_dict = mpi_comm.recv(source = i, tag = mpi_data_tag)

                self.img_cache_dict.update(img_cache_dict)

        return None


    def _mpi_cache_img_per_rank(self, idx_list):
        ''' Cache image in the seq_random_list unless a subset is specified
            using MPI.
        '''
        img_cache_dict = {}
        for idx in idx_list:
            # Skip those have been recorded...
            if idx in img_cache_dict: continue

            # Otherwise, record it...
            img = self.get_img(idx, verbose = True)
            img_cache_dict[idx] = img

        return img_cache_dict
