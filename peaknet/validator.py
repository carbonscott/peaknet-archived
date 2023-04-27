#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import torch
import tqdm
import numpy as np

from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast, GradScaler

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
    def __init__(self, model, dataset_test, config):
        self.model        = model
        self.dataset_test = dataset_test
        self.config  = config

        # Is GPU avaiable???
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        # Load checkpoint???
        if self.config.path_chkpt is not None:
            chkpt       = torch.load(self.config.path_chkpt)
            self.model.load_state_dict(chkpt)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        return None


    def validate(self, epoch = None, returns_loss = False, logs_batch_loss = False):
        """ The testing loop.  """

        # Load model and testing configuration...
        model, config = self.model, self.config

        if config.uses_mixed_precision:
            scaler = GradScaler()

        # Validate an epoch...
        # Load model state...
        model.eval()
        dataset_test = self.dataset_test
        loader_test  = DataLoader( dataset_test, shuffle     = config.shuffle, 
                                                 pin_memory  = config.pin_memory, 
                                                 batch_size  = config.batch_size,
                                                 num_workers = config.num_workers )

        # Train each batch...
        losses_epoch = []
        batch = tqdm.tqdm(enumerate(loader_test), total = len(loader_test), disable = config.tqdm_disable)
        for step_id, entry in batch:
            losses_batch = []

            ## batch_img, batch_mask, batch_metadata = entry
            batch_img, batch_mask = entry
            batch_img  = batch_img.to (self.device, dtype=torch.float)
            batch_mask = batch_mask.to(self.device, dtype=torch.float)

            with torch.no_grad():
                if config.uses_mixed_precision:
                    with autocast():
                        _, _, loss = self.model.forward(batch_img, batch_mask)
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                else:
                    _, _, loss = self.model.forward(batch_img, batch_mask)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus

                loss_val = loss.item()
                losses_batch.append(loss_val)

            loss_batch_mean = np.mean(losses_batch)
            losses_epoch.append(loss_batch_mean)

            if logs_batch_loss:
                logger.info(f"MSG (gpu:{self.device}) - epoch {epoch}, batch {step_id:d}, loss {loss_batch_mean:.8f}")

        loss_epoch_mean = np.mean(losses_epoch)
        logger.info(f"MSG (gpu:{self.device}) - epoch {epoch}, loss mean {loss_epoch_mean:.8f}")

        return loss_epoch_mean if returns_loss else None
