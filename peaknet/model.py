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
        self.method     = config.method
        self.pos_weight = config.pos_weight * torch.ones([1])

        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight = self.pos_weight)
        self.Sigmoid = nn.Sigmoid()
        self.MSELoss = nn.MSELoss()


    def forward(self, batch_img, batch_mask):
        # Find the predicted mask...
        batch_mask_predicted = self.method.forward(batch_img)

        # Crop the target mask...
        size_y, size_x = batch_mask_predicted.shape[-2:]
        batch_mask_true = center_crop(batch_mask, size_y, size_x)

        # Calculate BCE loss...
        loss_bce = self.BCEWithLogitsLoss(batch_mask_predicted, batch_mask_true)

        loss = loss_bce

        return batch_mask_predicted, batch_mask_true, loss


    def configure_optimizers(self, config_train):
        optimizer = torch.optim.Adam(self.method.parameters(), lr = config_train.lr)

        return optimizer


    def calc_dice_loss(self, batch_mask_predicted, batch_mask_true, smooth = 1.0):
        # Calculate batch intersection...
        batch_mask_intersection = batch_mask_predicted * batch_mask_true
        batch_mask_intersection_val = batch_mask_intersection.sum(dim = (-2, -1))

        # Calculate the dice coefficient...
        batch_dice_coeff  = 2.0 * batch_mask_intersection_val
        batch_dice_coeff /= batch_mask_predicted.sum(dim = (-2, -1)) \
                          + batch_mask_true.sum(dim = (-2, -1))      \
                          + smooth

        return -batch_dice_coeff.mean()


    def calc_iou_loss(self, batch_mask_predicted, batch_mask_true, smooth = 1.0):
        # Calculate batch intersection...
        batch_mask_intersection = batch_mask_predicted * batch_mask_true
        batch_mask_intersection_sum = batch_mask_intersection.sum(dim = (-2, -1))

        # Calculate the iou score...
        batch_iou  = batch_mask_intersection_sum + 1.0
        batch_iou /= batch_mask_predicted.sum(dim = (-2, -1)) \
                   + batch_mask_true.sum(dim = (-2, -1))      \
                   - batch_mask_intersection_sum              \
                   + 1.0

        return -batch_iou.mean()




    def calc_focal_loss(self, inputs, targets, alpha = 0.8, gamma = 2.0, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = nn.Sigmoid()(inputs)

        #flatten label and prediction tensors
        ## inputs = inputs.view(-1)
        ## targets = targets.view(-1)
        inputs  = torch.flatten(inputs , start_dim = -2)
        targets = torch.flatten(targets, start_dim = -2)

        #first compute binary cross-entropy 
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        return focal_loss.mean()
