#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import logging
import os

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
        ## self.pos_weight = config.pos_weight
        ## self.weight_mse_loss = getattr(config, "weight_mse_loss", 0.0)

        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma

        ## # Convert numerical values into torch tensors...
        ## self.pos_weight = torch.tensor(self.pos_weight)


    def init_params(self, fl_chkpt = None):
        # Initialize weights or reuse weights from a timestamp...
        def init_weights(module):
            # Initialize conv2d with Kaiming method...
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data, nonlinearity = 'relu')

                # Set bias zero since batch norm is used...
                module.bias.data.zero_()

        if fl_chkpt is None:
            self.apply(init_weights)
        else:
            drc_cwd          = os.getcwd()
            DRCCHKPT         = "chkpts"
            prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
            path_chkpt_prev  = os.path.join(prefixpath_chkpt, fl_chkpt)
            self.load_state_dict(torch.load(path_chkpt_prev))


    def forward(self, batch_img, batch_mask):
        # Find the predicted mask...
        batch_fmap_predicted = self.method.forward(batch_img)

        # Crop the target mask...
        size_y, size_x = batch_fmap_predicted.shape[-2:]
        batch_mask_true = center_crop(batch_mask, size_y, size_x)

        ## loss_bce = self.calc_bce_with_logit_loss(batch_fmap_predicted, batch_mask_true)
        ## loss_mse = self.calc_mse_with_logit_loss(batch_fmap_predicted, batch_mask_true)

        ## loss = loss_bce

        loss_focal = self.calc_binary_focal_loss_with_logits(batch_fmap_predicted, batch_mask_true, alpha = self.focal_alpha, gamma = self.focal_gamma)
        loss = loss_focal

        return batch_fmap_predicted, batch_mask_true, loss


    def calc_mse_with_logit_loss(self, batch_fmap_predicted, batch_mask_true):
        MSELoss = nn.MSELoss()

        batch_mask_predicted = batch_fmap_predicted.sigmoid()

        loss = MSELoss(batch_mask_predicted, batch_mask_true)

        return loss


    def calc_bce_with_logit_loss(self, batch_fmap_predicted, batch_mask_true):
        ## # Calculate BCE loss...
        ## num_pos = (batch_mask_true == 1).sum()
        ## num_neg = (batch_mask_true == 0).sum()
        ## pos_weight = num_neg / num_pos
        BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight = self.pos_weight)
        loss = BCEWithLogitsLoss(batch_fmap_predicted, batch_mask_true)

        return loss


    def configure_optimizers(self, config_train):
        optimizer = torch.optim.Adam(self.method.parameters(), lr = config_train.lr)

        return optimizer


    def calc_dice_loss(self, batch_fmap_predicted, batch_mask_true, smooth = 1.0):
        # Calculate batch intersection...
        batch_mask_intersection = batch_fmap_predicted * batch_mask_true
        batch_mask_intersection_val = batch_mask_intersection.sum(dim = (-2, -1))

        # Calculate the dice coefficient...
        batch_dice_coeff  = 2.0 * batch_mask_intersection_val
        batch_dice_coeff /= batch_fmap_predicted.sum(dim = (-2, -1)) \
                          + batch_mask_true.sum(dim = (-2, -1))      \
                          + smooth

        return -batch_dice_coeff.mean()


    def calc_iou_loss(self, batch_fmap_predicted, batch_mask_true, smooth = 1.0):
        # Calculate batch intersection...
        batch_mask_intersection = batch_fmap_predicted * batch_mask_true
        batch_mask_intersection_sum = batch_mask_intersection.sum(dim = (-2, -1))

        # Calculate the iou score...
        batch_iou  = batch_mask_intersection_sum + 1.0
        batch_iou /= batch_fmap_predicted.sum(dim = (-2, -1)) \
                   + batch_mask_true.sum(dim = (-2, -1))      \
                   - batch_mask_intersection_sum              \
                   + 1.0

        return -batch_iou.mean()


    def calc_binary_focal_loss_with_logits(self, x, y, alpha = 0.8, gamma = 2.0):
        '''
        Formula 5 from
        Lin, Tsung-Yi, Priya Goyal, Ross Girshick, Kaiming He, and Piotr
        Dollár. “Focal Loss for Dense Object Detection.” arXiv, February 7,
        2018. http://arxiv.org/abs/1708.02002.

        Also, the logit calculation that doesn't explode in gradients.

        Firstly, logit = -log(1 / (1+exp(-x)))
                       = log(1 + exp(-x))

        Then,
        ~~~math1
        simplify:
        log(1 + exp(-x)) = log(exp(0) + exp(-x))

        trick: 
        log(exp(x1) + exp(x2)) = a + log(exp(x1-a) + exp(x2-a)), where a = max(x1, x2)

        so:
        log(exp(0) + exp(-x)) = log(exp(x1) + exp(x2)), where x1 = 0 and x2 = -x
        = a + log(exp(0-a) + exp(-x-a))

        conclusion:
        log(1 + exp(-x)) = a + log(exp(0-a) + exp(-x-a))
        ~~~

        Finally,
        logit = a + log( exp(-a) + exp(-x-a) ), where a = max(0, -x)

        Calculate focal loss,
        binary_focal_loss = alpha * y * (1 - p)**gamma * logit +
                            (1 - y) * p**gamma * x             +
                            p**gamma * (1 - y) * logit
        '''
        # Calculate logit in a numerically stable way (see the docstring)...
        a = (-x).clamp(min = 0)
        logit = a + ( (-a).exp() + (-x-a).exp() ).log()

        # Derive probability based on logits...
        p = (-logit).exp()

        # Calculate the binary focal loss...
        binary_focal_loss = alpha * y * (1 - p)**gamma * logit + \
                            (1 - y) * p**gamma * x             + \
                            p**gamma * (1 - y) * logit

        return binary_focal_loss.mean()
