#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import random
import torch
import numpy as np
import tqdm
import skimage.measure as sm

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return None




class EpochManager:

    def __init__(self, trainer, validator, max_epochs = 1):
        self.trainer    = trainer
        self.validator  = validator
        self.max_epochs = max_epochs

        return None


    def run(self):
        # Track the min of loss from inf...
        loss_min = float('inf')

        # Start trainig and validation...
        for epoch in tqdm.tqdm(range(self.max_epochs)):
            # Run one epoch of training...
            self.trainer.train(is_save_checkpoint = False, epoch = epoch)

            # Pass the model to validator for immediate validation...
            self.validator.model = self.trainer.model

            # Run one epoch of training...
            loss_validate = self.validator.validate(is_return_loss = True, epoch = epoch)

            # Save checkpoint whenever validation loss gets smaller...
            # Notice it doesn't imply early stopping
            if loss_validate < loss_min: 
                # Save a checkpoint file...
                self.trainer.save_checkpoint()

                # Update the new loss...
                loss_min = loss_validate

        return None




class ConvVolume:
    """ Derive the output size of a conv net. """

    def __init__(self, size_y, size_x, channels, conv_dict):
        self.size_y      = size_y
        self.size_x      = size_x
        self.channels    = channels
        self.conv_dict   = conv_dict
        self.method_dict = { 'conv' : self._get_shape_from_conv2d, 
                             'pool' : self._get_shape_from_pool    }

        return None


    def shape(self):
        for layer_name in self.conv_dict["order"]:
            # Obtain the method name...
            method, _ = layer_name.split()

            # Unpack layer params...
            layer_params = self.conv_dict[layer_name]

            #  Obtain the size of the new volume...
            self.channels, self.size_y, self.size_x = \
                self.method_dict[method](**layer_params)

        return self.channels, self.size_y, self.size_x


    def _get_shape_from_conv2d(self, **kwargs):
        """ Returns the dimension of the output volumne. """
        size_y       = self.size_y
        size_x       = self.size_x
        out_channels = kwargs["out_channels"]
        kernel_size  = kwargs["kernel_size"]
        stride       = kwargs["stride"]
        padding      = kwargs["padding"]

        out_size_y = (size_y - kernel_size + 2 * padding) // stride + 1
        out_size_x = (size_x - kernel_size + 2 * padding) // stride + 1

        return out_channels, out_size_y, out_size_x


    def _get_shape_from_pool(self, **kwargs):
        """ Return the dimension of the output volumen. """
        size_y       = self.size_y
        size_x       = self.size_x
        out_channels = self.channels
        kernel_size  = kwargs["kernel_size"]
        stride       = kwargs["stride"]

        out_size_y = (size_y - kernel_size) // stride + 1
        out_size_x = (size_x - kernel_size) // stride + 1

        return out_channels, out_size_y, out_size_x




class MetaLog:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items(): setattr(self, k, v)


    def report(self):
        logger.info(f"___/ MetaLog \___")
        for k, v in self.__dict__.items():
            if k == 'kwargs': continue
            logger.info(f"KV - {k:16s} : {v}")




def downsample(assem, bin_row=2, bin_col=2, mask=None):
    """ Downsample an SPI image.  
        Adopted from https://github.com/chuckie82/DeepProjection/blob/master/DeepProjection/utils.py
    """
    if mask is None:
        combinedMask = np.ones_like(assem)
    else:
        combinedMask = mask
    downCalib  = sm.block_reduce(assem       , block_size=(bin_row, bin_col), func=np.sum)
    downWeight = sm.block_reduce(combinedMask, block_size=(bin_row, bin_col), func=np.sum)
    warr       = np.zeros_like(downCalib, dtype='float32')
    ind        = np.where(downWeight > 0)
    warr[ind]  = downCalib[ind] / downWeight[ind]

    return warr




def read_log(file):
    '''Return all lines in the user supplied parameter file without comments.
    '''
    # Retrieve key-value information...
    kw_kv     = "KV - "
    kv_dict   = {}

    # Retrieve data information...
    kw_data   = "DATA - "
    data_dict = {}
    with open(file,'r') as fh:
        for line in fh.readlines():
            # Collect kv information...
            if kw_kv in line:
                info = line[line.rfind(kw_kv) + len(kw_kv):]
                k, v = info.split(":", maxsplit = 1)
                if not k in kv_dict: kv_dict[k.strip()] = v.strip()

            # Collect data information...
            if kw_data in line:
                info = line[line.rfind(kw_data) + len(kw_data):]
                k = tuple( info.strip().split(",") )
                if not k in data_dict: data_dict[k] = True

    ret_dict = { "kv" : kv_dict, "data" : tuple(data_dict.keys()) }

    return ret_dict


def calc_dmat(emb1_list, emb2_list, is_sqrt = True):
    ''' Return a 2D distance matrix.

        emb1.shape: len(emb1_list), len(emb1_list[0])
        emb2.shape: len(emb2_list), len(emb2_list[0])
    '''
    # Calculate the difference vector...
    # emb1[:, None] has a dim of [ num1, 1   , dim ], equivalent to [num1, num2, dim] by stretching/replicating axis=1 num2 times.  
    # emb2[None, :] has a dim of [ 1   , num2, dim ], equivalent to [num1, num2, dim] by stretching/replicating axis=0 num2 times.  
    # subtraction returns dim of [ num1, num2, dim ]
    delta_distance_vector = emb1_list[:, None] - emb2_list[None, :]

    # Calculate the squared distance matrix...
    dmat = torch.sum( delta_distance_vector * delta_distance_vector, dim = -1 )

    # Apply square-root is necessary???
    if is_sqrt: dmat = torch.sqrt(dmat)

    return dmat




def split_dataset(dataset, fracA):
    ''' Split a dataset into two subsets A and B by user-specified fraction.
    '''
    # Get the size of the dataset and the subset A...
    size_dataset = len(dataset)
    size_fracA   = int(fracA * size_dataset)

    # Randomly choosing examples for constructing subset A...
    fracA_list = random.sample(dataset, size_fracA)

    # Obtain the subset B...
    fracB_list = set(dataset) - set(fracA_list)
    fracB_list = sorted(list(fracB_list))

    return fracA_list, fracB_list
