#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import torch
import tqdm
import numpy as np

from scipy import ndimage
from torch.utils.data.dataloader import DataLoader

from peaknet.datasets.transform import center_crop, coord_img_to_crop, coord_crop_to_img

logger = logging.getLogger(__name__)

class ConfigPredictor:
    path_chkpt   = None
    num_workers  = 1
    batch_size   = 1
    max_epochs   = 1
    lr           = 3e-04
    tqdm_disable = False

    def __init__(self, **kwargs):
        logger.info(f"__/ Configure Validator \___")
        # Set values of attributes that are not known when obj is created...
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"KV - {k:16s} : {v}")




class Predictor:
    def __init__(self, model, dataset_test, config_test):
        self.model        = model
        self.dataset_test = dataset_test
        self.config_test  = config_test

        # Load data to gpus if available...
        self.device = 'cpu'
        if self.config_test.path_chkpt is not None and torch.cuda.is_available():
            self.device = torch.cuda.current_device()

            # Load only the underlying model (model.method) from a checkpoint...
            chkpt = torch.load(self.config_test.path_chkpt)
            self.model.load_state_dict(chkpt)
            self.model.method.to(self.device)
            ## self.model = torch.nn.DataParallel(self.model).to(self.device)

        return None


    def predict(self, is_return_loss = False, epoch = None):
        """ The testing loop.  """

        # Load model and testing configuration...
        model, config_test = self.model, self.config_test

        # Validate an epoch...
        # Load model state...
        model.method.eval()
        dataset_test = self.dataset_test
        loader_test  = DataLoader( dataset_test, shuffle     = config_test.shuffle, 
                                                 pin_memory  = config_test.pin_memory, 
                                                 batch_size  = config_test.batch_size,
                                                 num_workers = config_test.num_workers )

        # Train each batch...
        losses_epoch = []
        batch = tqdm.tqdm(enumerate(loader_test), total = len(loader_test), disable = config_test.tqdm_disable)
        for step_id, entry in batch:
            # Fetch (batch, panel, dim_img_y, dim_img_x)...
            batch_size, panel_num, size_y, size_x = entry.shape

            # Combine batch and panel dim for model inference, but create a fake channel dim (dim = 1) ...
            batch_panel = entry.reshape(batch_size * panel_num, 1, size_y, size_x)
            #                                                   ^
            # Fake channel dim _________________________________|
            # (to facilitate the underlying model inference)

            ## import pdb; pdb.set_trace()

            # Load it to GPU
            batch_panel = batch_panel.to(self.device, dtype=torch.float32)

            # Predict...
            with torch.no_grad():
                batch_mask_predicted = self.model.method.forward(batch_panel)

            # Convert mask to coordinates...
            peak_list = self.convert_mask_to_coord(batch_panel, batch_mask_predicted)

            # [TODO] Maybe do some IO stuff to write peak

        return None


    def convert_mask_to_coord(self, batch_img, batch_mask_predicted):
        # Fetch the offset for coordinate conversion downstream...
        size_y_crop, size_x_crop = batch_mask_predicted.shape[-2:]
        _, offset_tuple = center_crop(batch_img, size_y_crop, size_x_crop, return_offset_ok = True)

        # Save them to cpu...
        batch_mask_predicted = batch_mask_predicted.cpu().detach().numpy()

        # Sequeeze the fake channel dimension (dim = 1)...
        batch_mask_predicted = np.squeeze(batch_mask_predicted, axis = 1)

        # Thresholding...
        adu_threshold = self.dataset_test.adu_threshold
        batch_mask = batch_mask_predicted.copy()
        batch_mask[  batch_mask_predicted < adu_threshold ] = 0

        # Put box on peaks...
        # Dim: (panel, y, x)
        batch_peak_list = self.calc_batch_center_of_mass(batch_mask)

        # Convert coordinates...
        # It tooks 0.1ms to run, not a priorty for optimization yet.
        batch_peak_list_converted = []
        size_y_img, size_x_img = batch_img.shape[-2:]
        for panel, y, x in batch_peak_list:
            if np.isnan(y): continue

            y_img, x_img = coord_crop_to_img((y, x), 
                                             (size_y_img, size_x_img), 
                                             (size_y_crop, size_x_crop),
                                             offset_tuple)
            batch_peak_list_converted.append((panel, y_img, x_img))

        return batch_peak_list_converted


    def calc_batch_center_of_mass(self, batch_mask):
        # Set up structure to find connected component in 2D only...
        structure = np.zeros((3, 3, 3))
        #                     ^  ^^^^
        # batch_______________|   |
        #                         |
        # 2D image________________|

        # Define structure in 2D image at the middle layer
        structure[1] = np.array([[0,1,0],
                                 [1,1,1],
                                 [0,1,0]])

        # Fetch labels...
        batch_label, batch_num_feature = ndimage.label(batch_mask, structure = structure)

        # Calculate batch center of mass...
        batch_center_of_mass = ndimage.center_of_mass(batch_mask, batch_label, range(batch_num_feature))

        return batch_center_of_mass
