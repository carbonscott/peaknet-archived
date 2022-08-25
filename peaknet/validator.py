#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import torch
import tqdm
import numpy as np

from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

class ConfigValidator:
    path_chkpt   = None
    num_workers  = 4
    batch_size   = 64
    max_epochs   = 10
    lr           = 0.001
    tqdm_disable = False

    def __init__(self, **kwargs):
        logger.info(f"__/ Configure Validator \___")
        # Set values of attributes that are not known when obj is created...
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"KV - {k:16s} : {v}")




class LossValidator:
    def __init__(self, model, dataset_test, config_test):
        self.model        = model
        self.dataset_test = dataset_test
        self.config_test  = config_test

        # Load data to gpus if available...
        self.device = 'cpu'
        if self.config_test.path_chkpt is not None and torch.cuda.is_available():
            self.device = torch.cuda.current_device()

            chkpt = torch.load(self.config_test.path_chkpt)
            self.model.load_state_dict(chkpt)
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        return None


    def validate(self, is_return_loss = False, epoch = None):
        """ The testing loop.  """

        # Load model and testing configuration...
        model, config_test = self.model, self.config_test

        # Validate an epoch...
        # Load model state...
        model.eval()
        dataset_test = self.dataset_test
        loader_test  = DataLoader( dataset_test, shuffle     = config_test.shuffle, 
                                                 pin_memory  = config_test.pin_memory, 
                                                 batch_size  = config_test.batch_size,
                                                 num_workers = config_test.num_workers )

        # Train each batch...
        losses_epoch = []
        batch = tqdm.tqdm(enumerate(loader_test), total = len(loader_test), disable = config_test.tqdm_disable)
        for step_id, entry in batch:
            losses_batch = []

            ## batch_img, batch_mask, batch_metadata = entry
            batch_img, batch_mask = entry
            batch_img  = batch_img.to (self.device)
            batch_mask = batch_mask.to(self.device)

            with torch.no_grad():
                _, _, loss = self.model.forward(batch_img, batch_mask)
                loss_val = loss.cpu().detach().numpy()
                losses_batch.append(loss_val)

            loss_batch_mean = np.mean(losses_batch)
            logger.info(f"MSG - epoch {epoch}, batch {step_id:d}, loss {loss_batch_mean:.8f}")
            losses_epoch.append(loss_batch_mean)

        loss_epoch_mean = np.mean(losses_epoch)
        logger.info(f"MSG - epoch {epoch}, loss mean {loss_epoch_mean:.8f}")

        return loss_epoch_mean if is_return_loss else None
