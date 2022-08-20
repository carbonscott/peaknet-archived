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

            batch_img, batch_mask, batch_metadata = entry
            batch_img = batch_img.to(self.device)

            with torch.no_grad():
                loss = self.model.forward(batch_img, batch_mask)
                loss_val = loss.cpu().detach().numpy()
                losses_batch.append(loss_val)

            loss_batch_mean = np.mean(losses_batch)
            logger.info(f"MSG - epoch {epoch}, batch {step_id:d}, loss {loss_batch_mean:.8f}")
            losses_epoch.append(loss_batch_mean)

        loss_epoch_mean = np.mean(losses_epoch)
        logger.info(f"MSG - epoch {epoch}, loss mean {loss_epoch_mean:.8f}")

        return loss_epoch_mean if is_return_loss else None




class PairValidator:
    def __init__(self, model, dataset_test, config_test):
        self.model        = model
        self.dataset_test = dataset_test
        self.config_test  = config_test

        # Load data to gpus if available
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

            chkpt = torch.load(self.config_test.path_chkpt)
            self.model.load_state_dict(chkpt)
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        return None


    def validate(self):
        """ The testing loop.  """

        # Load model and testing configuration...
        model, config_test = self.model, self.config_test

        # Train an epoch...
        # Load model state
        model.eval()
        dataset_test = self.dataset_test
        loader_test  = DataLoader( dataset_test, shuffle     = config_test.shuffle, 
                                                 pin_memory  = config_test.pin_memory, 
                                                 batch_size  = config_test.batch_size,
                                                 num_workers = config_test.num_workers )

        # Train each batch...
        batch = tqdm.tqdm(enumerate(loader_test), total = len(loader_test))
        for step_id, entry in batch:
            rmsds = []

            img_anchor, img_second, label_anchor, title_anchor, title_second = entry

            img_anchor = img_anchor.to(self.device)
            img_second = img_second.to(self.device)

            with torch.no_grad():
                # Look at each example in a batch...
                for i in range(len(label_anchor)):
                    # Biolerplate unsqueeze due to the design of PyTorch Conv2d, no idea how to improve yet...
                    if config_test.isflat:
                        _, _, rmsd = self.model.forward(img_anchor[i], img_second[i])
                    else:
                        ## print(title_anchor[i], title_second[i])
                        _, _, rmsd = self.model.forward(img_anchor[i].unsqueeze(0), img_second[i].unsqueeze(0))

                    rmsd_val = rmsd.cpu().detach().numpy()
                    rmsds.append(rmsd_val)
                    logger.info(f"DATA - {title_anchor[i]}, {title_second[i]}, {rmsd_val:12.8f}")




class MultiwayQueryValidator:
    def __init__(self, model, dataset_test, config_test):
        self.model        = model
        self.dataset_test = dataset_test
        self.config_test  = config_test

        # Load data to gpus if available
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

            chkpt = torch.load(self.config_test.path_chkpt)
            self.model.load_state_dict(chkpt)
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        return None


    def validate(self):
        """ The testing loop.  """

        # Load model and testing configuration...
        model, config_test = self.model, self.config_test

        # Train an epoch...
        # Load model state
        model.eval()
        dataset_test = self.dataset_test
        loader_test  = DataLoader( dataset_test, shuffle     = config_test.shuffle, 
                                                 pin_memory  = config_test.pin_memory, 
                                                 batch_size  = config_test.batch_size,
                                                 num_workers = config_test.num_workers )

        # Train each batch...
        batch = tqdm.tqdm(enumerate(loader_test), total = len(loader_test))
        for step_id, entry in batch:
            # Unpack entry...
            # Not a good design, but it's okay for now (03/03/2020)
            # Shape of entry: (unpack_dim, batch, internal_shape)
            # Internal shape of image is 2d, string is 1d
            batch_img_list = entry[                 : len(entry) // 2 ]
            batch_str_list = entry[ len(entry) // 2 :                 ]

            # Assign returned subcategory for img and str...
            # batch_img_query : (num_query = 1     , size_batch, size_image2d)
            # batch_img_test  : (num_test_per_query, size_batch, size_image2d)
            # batch_str_query : (num_query = 1     , size_batch, size_str)
            # batch_str_test  : (num_test_per_query, size_batch, size_str)
            batch_img_query, batch_img_test = batch_img_list[0:1], batch_img_list[1:]
            batch_str_query, batch_str_test = batch_str_list[0:1], batch_str_list[1:]

            # Load imgs to gpu...
            batch_img_query = torch.stack(batch_img_query).to(self.device)
            batch_img_test  = torch.stack(batch_img_test).to(self.device)

            # Calculate the squared distance between embeddings...
            # (size_batch, size_image) => (size_batch, len_emb)
            with torch.no_grad():
                # batch_emb_query : (num_test_per_query, size_batch, len_emb)
                # batch_dist      : (num_test_per_query, size_batch         )
                batch_emb_query, _, batch_dist = self.model.forward(batch_img_query, batch_img_test)
                batch_dist_numpy = batch_dist.cpu().detach().numpy()

            # Go through each item in a batch...
            num_test_per_query, size_batch = batch_img_test.shape[:2]
            for i in range(size_batch):
                # Go through each test against the query...
                msg = [ f"{batch_str_test[j][i]} : {batch_dist_numpy[j][i]:12.8f}" for j in range(num_test_per_query) ]

                # Return a line for each batch...
                log_header = f"DATA - {batch_str_query[0][i]}, "
                log_msg = log_header + ", ".join(msg)
                logger.info(log_msg)

        return None




class EmbeddingCalculator:
    def __init__(self, model, dataset_test, config_test):
        self.model        = model
        self.dataset_test = dataset_test
        self.config_test  = config_test

        # Load data to gpus if available
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

            if self.config_test.path_chkpt is not None:
                chkpt = torch.load(self.config_test.path_chkpt)
                self.model.load_state_dict(chkpt)
            else:
                # Initialize weights...
                def init_weights(module):
                    if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
                        module.weight.data.normal_(mean = 0.0, std = 0.02)
                self.model.apply(init_weights)
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        return None


    def run(self):
        """ The testing loop.  """

        # Load model and testing configuration...
        model, config_test = self.model, self.config_test

        # Train an epoch...
        # Load model state
        model.eval()
        dataset_test = self.dataset_test
        loader_test  = DataLoader( dataset_test, shuffle     = config_test.shuffle, 
                                                 pin_memory  = config_test.pin_memory, 
                                                 batch_size  = config_test.batch_size,
                                                 num_workers = config_test.num_workers )

        # Train each batch...
        batch = tqdm.tqdm(enumerate(loader_test), total = len(loader_test))
        counter_item = 0
        for batch_id, entry in batch:
            # Unpack entry...
            # Not a good design, but it's okay for now (03/03/2020)
            # Shape of entry: (unpack_dim, batch, internal_shape)
            # Internal shape of image is 2d, string is 1d
            batch_img = entry[0]
            batch_str = entry[1]

            # Load imgs to gpu...
            # batch_img : torch.Tensor, (size_batch, size_image2d)
            batch_img = batch_img.to(self.device)

            with torch.no_grad():
                # Calculate image embedding for this batch...
                # batch_emb : torch.Tensor, (size_batch, len_emb)
                batch_emb = self.model.forward(batch_img)

            # Save the embedding...
            size_batch, len_emb = batch_emb.shape
            if batch_id == 0:
                num_imgs  = len(dataset_test)
                dataset_emb      = torch.zeros(num_imgs, len_emb)
                rng_start = 0
            dataset_emb[rng_start : rng_start + size_batch] = batch_emb
            rng_start += size_batch

            # Logging...
            for i in range(size_batch):
                # Fetch the descriptor...
                msg = batch_str[i]

                # Return a line for each batch...
                log_header = f"DATA - {counter_item:06d} - "
                log_msg = log_header + msg
                logger.info(log_msg)

                counter_item += 1

        return dataset_emb
