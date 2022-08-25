#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import logging

from peaknet.datasets.transform import center_crop

logger = logging.getLogger(__name__)

class ConfigPeakFinderModel:

    def __init__(self, **kwargs):
        logger.info(f"___/ Configure Model \___")

        # Set values of attributes that are not known when obj is created
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"KV - {k:16} : {v}")


class PeakFinderModel(nn.Module):
    """ The peak finder model. """

    def __init__(self, config):
        super().__init__()
        self.method = config.method

        self.loss_func = nn.BCELoss()
        self.sigmoid   = nn.Sigmoid()


    def forward(self, batch_img, batch_mask):
        # Find the predicted mask...
        batch_mask_predicted = self.method.forward(batch_img)

        # Pass through sigmoid...
        batch_mask_predicted_with_sigmoid = self.sigmoid(batch_mask_predicted)

        # Crop the target mask...
        size_y, size_x = batch_mask_predicted_with_sigmoid.shape[-2:]
        batch_mask_crop = center_crop(batch_mask, size_y, size_x)

        # Calculate loss...
        loss = self.loss_func(batch_mask_predicted_with_sigmoid, batch_mask_crop)

        return batch_mask_predicted_with_sigmoid, batch_mask_crop, loss


    def configure_optimizers(self, config_train):
        optimizer = torch.optim.Adam(self.method.parameters(), lr = config_train.lr)

        return optimizer
