#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import random
import csv
import os
import h5py
import pickle
import logging

from scipy.spatial    import cKDTree
from torch.utils.data import Dataset

from peaknet.utils                  import set_seed, split_dataset
from peaknet.datasets.stream_parser import StreamParser, GeomInterpreter

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




class SFXPanelDataset(Dataset):
    """
    SFX images are collected from multiple datasets specified in the input csv
    file. All images are organized in a plain list.  

    get_     method returns data by sequential index.
    extract_ method returns object by files.
    """

    def __init__(self, config):
        self.fl_csv        = getattr(config, 'fl_csv'        , None)
        self.drc_project   = getattr(config, 'drc_project'   , None)
        self.size_sample   = getattr(config, 'size_sample'   , None)
        self.frac_train    = getattr(config, 'frac_train'    , None)    # Proportion/Fraction of training examples
        self.frac_validate = getattr(config, 'frac_validate' , None)    # Proportion/Fraction of validation examples
        self.dataset_usage = getattr(config, 'dataset_usage' , None)    # train, validate, test
        self.seed          = getattr(config, 'seed'          , None)
        self.dist          = getattr(config, 'dist'          , 5)
        self.trans         = getattr(config, 'trans'         , None)

        # Variables that capture raw information from the input (stream files)
        self.fl_stream_list     = []
        self.metadata_orig_list = [] # ...A list of (fl_cxi, event_crystfel) that have labeled peaks
        self.peak_orig_list     = [] # ...A list of labeled peaks
        self.stream_cache_dict  = {} # ...A dictionary of file to stream mapping
        self.is_cache           = False

        # Variables that capture information in data spliting
        self.metadata_list   = []
        self.peak_list       = []

        # Variables for caching...
        self.peak_cache_dict    = {} # ...A list of image data and their corresponding labels
        self.raw_img_cache_dict = {}
        self.raw_label_cache_dict = {}

        # Set the seed...
        # Debatable whether seed should be set in the dataset or in the running code
        if not self.seed is None: set_seed(self.seed)

        # Collect stream files from the csv...
        with open(self.fl_csv, 'r') as fh:
            lines = csv.reader(fh)

            next(lines)

            for line in lines:
                fl_stream = line[0]
                self.fl_stream_list.append(fl_stream)

        # Obtain all indexed SFX event from stream files...
        for fl_stream in self.fl_stream_list:
            # Get the stream object and save it to a global dictionary...
            stream_per_file_dict = self.parse_stream(fl_stream)
            if not fl_stream in self.stream_cache_dict:
                self.stream_cache_dict[fl_stream] = stream_per_file_dict

            # Create a list of data entry from a stream file...
            metadata_per_stream, peak_list_per_stream = \
                self.extract_metadata_and_labeled_peak_from_streamfile(fl_stream)

            # Accumulate metadata...
            self.metadata_orig_list.extend(metadata_per_stream)    # (fl_stream, fl_cxi, event_crystfel)

            # Accumulate all peaks...
            self.peak_orig_list.extend(peak_list_per_stream)

        # Split original dataset sequence into training sequence and holdout sequence...
        seq_orig_list = list(range(len(self.metadata_orig_list)))
        seq_train_list, seq_holdout_list = split_dataset(seq_orig_list, self.frac_train)

        # Calculate the percentage of validation in the whole holdout set...
        frac_holdout = 1.0 - self.frac_train
        frac_validate_in_holdout = self.frac_validate / frac_holdout if self.frac_validate is not None else 0.5

        # Split holdout dataset into validation and test...
        seq_valid_list, seq_test_list = split_dataset(seq_holdout_list, frac_validate_in_holdout)

        # Choose which dataset is going to be used, defaults to original set...
        dataset_by_usage_dict = {
            'train'    : seq_train_list,
            'validate' : seq_valid_list,
            'test'     : seq_test_list,
        }
        seq_random_list = seq_orig_list
        if self.dataset_usage in dataset_by_usage_dict:
            seq_random_list = dataset_by_usage_dict[self.dataset_usage]

        # Create data list based on the sequence...
        self.metadata_list = [ self.metadata_orig_list[i] for i in seq_random_list ]
        self.peak_list     = [ self.peak_orig_list[i]     for i in seq_random_list ]

        return None


    def parse_stream(self, fl_stream):
        # Initialize the object to return...
        stream_per_file_dict = None

        # Find the basename of the stream file...
        basename_stream = os.path.basename(fl_stream)
        basename_stream = basename_stream[:basename_stream.rfind('.')]

        # Check if a corresponding pickle file exists...
        fl_pickle         = f"{basename_stream}.pickle"
        prefix_pickle     = 'pickles'
        prefixpath_pickle = os.path.join(self.drc_project, prefix_pickle)
        if not os.path.exists(prefixpath_pickle): os.makedirs(prefixpath_pickle)
        path_pickle = os.path.join(prefixpath_pickle, fl_pickle)

        # Obtain key information from stream by loading the pickle file if it exists...
        if os.path.exists(path_pickle):
            with open(path_pickle, 'rb') as fh:
                stream_per_file_dict = pickle.load(fh)

        # Otherwise, parse the stream file...
        else:
            # Employ stream parser to extract key info from stream...
            stream_parser = StreamParser(fl_stream)
            stream_parser.parse()
            stream_per_file_dict = stream_parser.stream_per_file_dict

            # Save the stream result in a pickle file...
            with open(path_pickle, 'wb') as fh:
                pickle.dump(stream_per_file_dict, fh, protocol = pickle.HIGHEST_PROTOCOL)

        return stream_per_file_dict


    def get_raw_peak(self, idx):
        '''
        Return raw found peaks and indexed peaks in a tuple.
        '''
        # Read image...
        # Ignore which stream file this information is extracted from
        fl_stream, fl_cxi, event_crystfel, panel = self.metadata_list[idx]

        # Fetch stream either from scratch or from a cached dictionary...
        stream_per_file_dict = self.parse_stream(fl_stream) if not fl_stream in self.stream_cache_dict \
                                                            else self.stream_cache_dict[fl_stream]

        # Get the right chunk
        panel_dict = stream_per_file_dict['chunk'][fl_cxi][event_crystfel]

        # Find all peaks...
        peak_saved_dict        = panel_dict[panel]
        found_per_panel_dict   = peak_saved_dict['found']
        indexed_per_panel_dict = peak_saved_dict['indexed']

        return found_per_panel_dict, indexed_per_panel_dict


    def extract_metadata_and_labeled_peak_from_streamfile(self, fl_stream):
        # Fetch stream either from scratch or from a cached dictionary...
        stream_per_file_dict = self.parse_stream(fl_stream) if not fl_stream in self.stream_cache_dict \
                                                            else self.stream_cache_dict[fl_stream]

        # Get the chunk...
        chunk_dict = stream_per_file_dict['chunk']

        # Work on each cxi file...
        metadata_list = []
        peak_list     = []
        for fl_cxi, event_crystfel_list in chunk_dict.items():
            # Only keep those both found and indexed peaks...
            for event_crystfel in event_crystfel_list:
                # Get panels in an event...
                panel_dict = chunk_dict[fl_cxi][event_crystfel]

                # Find all peaks...
                found_per_event_dict   = {}
                indexed_per_event_dict = {}
                for panel, peak_saved_dict in panel_dict.items():
                    found_per_event_dict[panel]   = peak_saved_dict['found']
                    indexed_per_event_dict[panel] = peak_saved_dict['indexed']

                # Filter peaks and only keep those that are indexed...
                for panel, found_list in found_per_event_dict.items():
                    panel_descriptor = (fl_stream, fl_cxi, event_crystfel, panel)

                    # Skp panels that have no peaks found...
                    if not panel in indexed_per_event_dict: continue

                    # Skp panel that have peaks found but nothing indexed...
                    indexed_list = indexed_per_event_dict[panel]
                    if len(indexed_list) == 0: continue

                    found_list_tree   = cKDTree(found_list)
                    indexed_list_tree = cKDTree(indexed_list)

                    # Filter based on a thresholding distance that tolerates the separation between a found peak and an indexed peak...
                    idx_indexed_peak_list = found_list_tree.query_ball_tree(indexed_list_tree, r = self.dist)

                    # Extract filtered peak positions for each panel...
                    peak_per_panel_list = []
                    for idx, neighbor_list in enumerate(idx_indexed_peak_list):
                        if len(neighbor_list) == 0: continue

                        x, y = found_list[idx]
                        x = int(x)
                        y = int(y)

                        peak_per_panel_list.append((x, y))

                    # Accumulate data for making a label...
                    metadata_list.append(panel_descriptor)
                    peak_list.append(peak_per_panel_list)

        return metadata_list, peak_list


    def __len__(self):
        return len(self.seq_random_list)


    def cache_img(self, idx_list = []):
        ''' Cache image in the seq_random_list unless a subset is specified.
        '''
        # If subset is not give, then go through the whole set...
        if not len(idx_list): idx_list = range(len(self.metadata_list))

        for idx in idx_list:
            # Skip those have been recorded...
            if idx in self.peak_cache_dict: continue

            # Otherwise, record it
            img, label = self.get_img_and_label(idx, verbose = True)
            self.peak_cache_dict[idx] = (img, label)

        return None


    def get_img_and_label(self, idx, verbose = False):
        # Unpack the descriptor to locate a panel...
        fl_stream, fl_cxi, event_crystfel, panel = self.metadata_list[idx]

        # Get geom information...
        stream_dict       = self.stream_cache_dict[fl_stream]
        geom_dict         = stream_dict['geom']
        cheetah_geom_dict = GeomInterpreter(geom_dict).interpret()

        # Load an image...
        # From scratch and cache it
        cache_key = (fl_stream, fl_cxi, event_crystfel)
        if not cache_key in self.raw_img_cache_dict:
            with h5py.File(fl_cxi, 'r') as fh:
                raw_img = fh["/entry_1/instrument_1/detector_1/data"][event_crystfel]

                self.raw_img_cache_dict[cache_key] = raw_img

        # Otherwise, load it from cache
        raw_img = self.raw_img_cache_dict[cache_key]

        # Select an area as a panel image and a masked panel...
        x_min, y_min, x_max, y_max = cheetah_geom_dict[panel]
        panel_img = raw_img[y_min : y_max, x_min : x_max]

        # Create a mask that works as the label...
        panel_label         = np.zeros_like(panel_img)    # ...A mask
        offset              = 4
        size_y, size_x      = panel_img.shape
        peak_per_panel_list = self.peak_list[idx]
        for peak in peak_per_panel_list:
            # Unack coordiante...
            x, y  = peak

            # Get local coordinate for the panel...
            x = x - x_min
            y = y - y_min

            # Find a patch around a peak with global image coordinate...
            patch_x_min = max(int(x - offset), 0)
            patch_x_max = min(int(x + offset), size_x)
            patch_y_min = max(int(y - offset), 0)
            patch_y_max = min(int(y + offset), size_y)
            patch_img = panel_img[patch_y_min : patch_y_max, patch_x_min : patch_x_max]

            # Figure out the local threshold...
            std_level  = 1
            patch_mean = np.mean(patch_img)
            patch_std  = np.std (patch_img)
            threshold  = patch_mean + std_level * patch_std

            # Mask the patch area out...
            panel_label[patch_y_min : patch_y_max, patch_x_min : patch_x_max][~(patch_img < threshold)] = 1.0

        if verbose: logger.info(f'DATA LOADING - {fl_cxi} {event_crystfel} {panel}.')

        return panel_img, panel_label


    def __getitem__(self, idx):
        img, label = self.peak_cache_dict[idx] if   idx in self.peak_cache_dict \
                                               else self.get_img_and_label(idx)

        # Apply any possible transformation...
        # How to define a custom transform function?
        # Input : img, **kwargs 
        # Output: img_transfromed
        if self.trans is not None:
            img = self.trans(img)

        # Normalize input image...
        img_mean = np.mean(img)
        img_std  = np.std(img)
        img_norm = (img - img_mean) / img_std

        return img_norm, label




class SFXPanelDatasetMini(SFXPanelDataset):
    def __init__(self, config):
        super().__init__(config)

        self.form_miniset()


    def __len__(self):
        return self.size_sample


    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)

        return img, label


    def form_miniset(self):
        size_sample = self.size_sample

        # Draw a miniset (subset)...
        idx_metadata_list = range(len(self.metadata_list))
        idx_metadata_miniset = random.sample(idx_metadata_list, k = size_sample)

        # Reconstruct metadata_list and peak_list...
        self.metadata_list = [ self.metadata_list[i] for i in idx_metadata_miniset ]
        self.peak_list     = [ self.peak_list[i]     for i in idx_metadata_miniset ]
