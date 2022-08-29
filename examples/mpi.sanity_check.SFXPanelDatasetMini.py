#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
from peaknet.datasets.panels        import ConfigDataset, SFXPanelDatasetMini
from peaknet.methods.unet           import UNet
from peaknet.model                  import ConfigPeakFinderModel, PeakFinderModel
from peaknet.datasets.stream_parser import GeomInterpreter
from peaknet.datasets.transform     import center_crop, coord_img_to_crop

import matplotlib              as mpl
import matplotlib.pyplot       as plt
import matplotlib.colors       as mcolors
import matplotlib.patches      as mpatches
import matplotlib.transforms   as mtransforms
import matplotlib.font_manager as font_manager

from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()

class SanityChecker:
    def __init__(self, model, dataset, path_chkpt, figsize, **kwargs):
        self.model          = model
        self.dataset        = dataset
        self.path_chkpt = path_chkpt
        self.figsize        = figsize

        for k, v in kwargs.items(): setattr(self, k, v)

        # Load model to gpus if available...
        self.device = 'cpu'
        if self.path_chkpt is not None and torch.cuda.is_available():
            self.device = torch.cuda.current_device()

            chkpt = torch.load(self.path_chkpt)
            self.model.load_state_dict(chkpt)
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        self.config_fonts()
        ## self.config_colorbar()

        ## self.linewidth = 0.1
        self.linewidth = 1.0

        self.b_offset = 2

        return None


    def fetch_data(self, idx):
        # Load model and testing configuration...
        model = self.model

        # Load model state...
        model.eval()
        dataset = self.dataset

        # Get the real image and mask...
        fl_stream, fl_cxi, event_crysfel, panel = self.dataset.metadata_list[idx]
        img, mask = dataset.get_img_and_label(idx)

        # Add a fake batch dim and move the data to gpu if available...
        batch_img  = torch.Tensor(img [None,])
        batch_mask = torch.Tensor(mask[None,])
        batch_img  = batch_img.to(self.device)
        batch_mask = batch_mask.to(self.device)

        # Find the predicted mask...
        batch_mask_predicted, batch_mask_true, loss = model.forward(batch_img, batch_mask)

        # Crop the original image...
        size_y, size_x = batch_mask_predicted.shape[-2:]
        batch_img_true = center_crop(batch_img, size_y, size_x)
        batch_img_true = batch_img_true.cpu().detach().numpy()
        img_true = batch_img_true.reshape(*batch_img_true.shape[-3:])    # Remove fake batch layer

        # Save them to cpu...
        batch_mask_predicted = batch_mask_predicted.cpu().detach().numpy()
        batch_mask_true      = batch_mask_true.cpu().detach().numpy()
        mask_predicted = batch_mask_predicted.reshape(*batch_mask_predicted.shape[-3:])    # Remove fake batch layer
        mask_true      = batch_mask_true.reshape(*batch_mask_true.shape[-3:])

        # Get the raw image...
        raw_img_key = (fl_stream, fl_cxi, event_crysfel)
        raw_img     = dataset.raw_img_cache_dict[raw_img_key]

        # Get geometry content...
        stream_dict       = dataset.stream_cache_dict[fl_stream]
        geom_dict         = stream_dict['geom']
        cheetah_geom_dict = GeomInterpreter(geom_dict).interpret()

        # Get peaks...
        found_list, indexed_list = dataset.get_raw_peak(idx)
        peak_list = dataset.peak_list[idx]

        if self.dataset.add_channel_ok:
            raw_img        = raw_img       [0]
            img            = img           [0]
            mask           = mask          [0]
            mask_predicted = mask_predicted[0]
            mask_true      = mask_true     [0]
            img_true       = img_true      [0]

        # Sync for visualization...
        self.fl_stream         = fl_stream
        self.fl_cxi            = fl_cxi
        self.event_crysfel     = event_crysfel
        self.panel             = panel
        self.cheetah_geom_dict = cheetah_geom_dict
        self.raw_img           = raw_img
        self.img               = img
        self.mask              = mask
        self.found_list        = found_list
        self.indexed_list      = indexed_list
        self.peak_list         = peak_list
        self.mask_predicted    = mask_predicted
        self.mask_true         = mask_true
        self.img_true          = img_true
        self.loss              = loss


    def config_fonts(self):
        # Where to load external font...
        drc_py    = os.path.dirname(os.path.realpath(__file__))
        drc_font  = os.path.join("fonts", "Helvetica")
        fl_ttf    = f"Helvetica.ttf"
        path_font = os.path.join(drc_py, drc_font, fl_ttf)
        prop_font = font_manager.FontProperties( fname = path_font )

        # Add Font and configure font properties
        font_manager.fontManager.addfont(path_font)
        prop_font = font_manager.FontProperties(fname = path_font)
        self.prop_font = prop_font

        # Specify fonts for pyplot...
        plt.rcParams['font.family'] = prop_font.get_name()
        plt.rcParams['font.size']   = 12

        return None


    def create_panels(self):
        ncols = 5
        nrows = 2

        fig   = plt.figure(figsize = self.figsize)
        gspec = fig.add_gridspec( nrows, ncols,
                                  width_ratios  = [1/2, 1/2, 1/2, 1/2, 1/20],
                                  ## height_ratios = [4/20, 4/20, 4/20, 4/20, 4/20], 
                                )
        ax_list = [ fig.add_subplot(gspec[0:2, 0:2], aspect = 1) ]
        ax_list.append( fig.add_subplot(gspec[0, 2], aspect = 1) )
        ax_list.append( fig.add_subplot(gspec[0, 3], aspect = 1) )
        ax_list.append( fig.add_subplot(gspec[1, 2], aspect = 1) )
        ax_list.append( fig.add_subplot(gspec[1, 3], aspect = 1) )
        ax_list.append( fig.add_subplot(gspec[:, 4], aspect = 1) )

        self.ncols = ncols
        self.nrows = nrows

        return fig, ax_list


    def config_colorbar(self, vmin = -1, vcenter = 0, vmax = 1):
        # Plot image...
        self.divnorm = mcolors.TwoSlopeNorm(vcenter = vcenter, vmin = vmin, vmax = vmax)


    def plot_geom(self):
        ax_img  = self.ax_list[0]

        cheetah_geom_dict = self.cheetah_geom_dict

        y_bmin, x_bmin, y_bmax, x_bmax = 0, 0, 0, 0
        for k, (x_min, y_min, x_max, y_max) in cheetah_geom_dict.items():
            w = x_max - x_min
            h = y_max - y_min
            rec = mpatches.Rectangle(xy = (x_min, y_min), width = w, height = h, facecolor='none', edgecolor = 'green')

            ax_img.add_patch(rec)

            y_text = (y_min + y_max) / 2
            x_text = (x_min + x_max) / 2
            ax_img.text(x = x_text, y = y_text, s = k, 
                        color = 'yellow',
                        horizontalalignment = 'center',
                        verticalalignment   = 'center',
                        transform=ax_img.transData)

            y_bmin = min(y_bmin, y_min)
            x_bmin = min(x_bmin, x_min)
            y_bmax = max(y_bmax, y_max)
            x_bmax = max(x_bmax, x_max)

        offset = 10
        ax_img.set_xlim([x_bmin - offset, x_bmax + offset])
        ax_img.set_ylim([y_bmin - offset, y_bmax + offset])

        return None


    def plot_raw_img(self):
        img = self.raw_img

        ax_img  = self.ax_list[0]
        ax_cbar = self.ax_list[-1]
        ax_cbar.set_aspect('auto')

        std_level = 4
        img_mean = np.mean(img)
        img_std  = np.std(img)
        threshold = img_mean + img_std * std_level
        ## img[img < threshold] = 0.0
        ## img[img >= threshold] = 1.0
        self.threshold = threshold
        vmin = 0
        vmax = threshold

        im = ax_img.imshow(img, vmin = vmin, vmax = vmax)
        im.set_cmap('gray')
        plt.colorbar(im, cax = ax_cbar, orientation="vertical", pad = 0.05)

        return None


    def plot_img(self):
        img = self.img

        ax_img  = self.ax_list[1]
        ax_cbar = self.ax_list[-1]
        ax_cbar.set_aspect('auto')

        ## std_level = 4
        ## img_mean = np.mean(img)
        ## img_std  = np.std(img)
        ## threshold = img_mean + img_std * std_level
        ## ## img[img < threshold] = 0.0
        ## ## img[img >= threshold] = 1.0
        ## self.threshold = threshold
        vmin = 0
        vmax = self.threshold

        im = ax_img.imshow(img, vmin = vmin, vmax = vmax)
        im.set_cmap('gray')
        ## plt.colorbar(im, cax = ax_cbar, orientation="vertical", pad = 0.05)

        ax_img.invert_yaxis()

        size_y, size_x = img.shape[-2:]
        b_offset = self.b_offset
        y_bmin, x_bmin = 0, 0
        y_bmax, x_bmax = size_y, size_x
        ax_img.set_xlim([x_bmin - b_offset, x_bmax + b_offset])
        ax_img.set_ylim([y_bmin - b_offset, y_bmax + b_offset])


        return None


    def plot_found(self):
        found_list = self.found_list
        offset = 4

        ax_raw = self.ax_list[0]
        for x, y in found_list:
            x_bottom_left = int(x - offset)
            y_bottom_left = int(y - offset)

            rec_obj = mpatches.Rectangle((x_bottom_left, y_bottom_left), 
                                         2 * offset, 2 * offset, 
                                         linewidth = self.linewidth, 
                                         edgecolor = 'yellow', 
                                         facecolor='none')
            ax_raw.add_patch(rec_obj)

        ax_img = self.ax_list[1]
        x_min, y_min, x_max, y_max = self.cheetah_geom_dict[self.panel]
        for x, y in found_list:
            x -= x_min
            y -= y_min

            x_bottom_left = int(x - offset)
            y_bottom_left = int(y - offset)

            rec_obj = mpatches.Rectangle((x_bottom_left, y_bottom_left), 
                                         2 * offset, 2 * offset, 
                                         linewidth = self.linewidth, 
                                         edgecolor = 'yellow', 
                                         facecolor='none')
            ax_img.add_patch(rec_obj)

        # BAD CODING PRACTICE, not good for maintainence, change it later!!!
        ax_img = self.ax_list[3]

        img  = self.img
        mask = self.mask_predicted

        size_img_tuple  = img.shape[-2:]
        size_crop_tuple = mask.shape[-2:]

        x_min, y_min, x_max, y_max = self.cheetah_geom_dict[self.panel]
        for x, y in found_list:
            x -= x_min
            y -= y_min

            # Transform...
            coord_tuple = (y, x)
            y, x = coord_img_to_crop(coord_tuple, size_img_tuple, size_crop_tuple)

            x_bottom_left = int(x - offset)
            y_bottom_left = int(y - offset)

            rec_obj = mpatches.Rectangle((x_bottom_left, y_bottom_left), 
                                         2 * offset, 2 * offset, 
                                         linewidth = self.linewidth, 
                                         edgecolor = 'yellow', 
                                         facecolor='none')
            ax_img.add_patch(rec_obj)



    def plot_indexed(self):
        indexed_list = self.indexed_list
        offset = 4

        ax_raw = self.ax_list[0]
        for x, y in indexed_list:
            x_bottom_left = int(x - offset)
            y_bottom_left = int(y - offset)

            ## rec_obj = mpatches.Rectangle((x_bottom_left, y_bottom_left), 
            ##                              2 * offset, 2 * offset, 
            x = int(x)
            y = int(y)
            rec_obj = mpatches.Circle((x, y), offset,
                                         linewidth = self.linewidth, 
                                         edgecolor = 'cyan', 
                                         facecolor='none')
            ax_raw.add_patch(rec_obj)

        ax_img = self.ax_list[1]
        x_min, y_min, x_max, y_max = self.cheetah_geom_dict[self.panel]
        for x, y in indexed_list:
            x -= x_min
            y -= y_min

            x_bottom_left = int(x - offset)
            y_bottom_left = int(y - offset)

            ## rec_obj = mpatches.Rectangle((x_bottom_left, y_bottom_left), 
            ##                              2 * offset, 2 * offset, 
            x = int(x)
            y = int(y)
            rec_obj = mpatches.Circle((x, y), offset,
                                         linewidth = self.linewidth, 
                                         edgecolor = 'cyan', 
                                         facecolor='none')
            ax_img.add_patch(rec_obj)

        # BAD CODING PRACTICE, not good for maintainence, change it later!!!
        ax_img = self.ax_list[3]

        img  = self.img
        mask = self.mask_predicted

        size_img_tuple  = img.shape[-2:]
        size_crop_tuple = mask.shape[-2:]

        x_min, y_min, x_max, y_max = self.cheetah_geom_dict[self.panel]
        for x, y in indexed_list:
            x -= x_min
            y -= y_min

            # Transform...
            coord_tuple = (y, x)
            y, x = coord_img_to_crop(coord_tuple, size_img_tuple, size_crop_tuple)

            x_bottom_left = int(x - offset)
            y_bottom_left = int(y - offset)

            ## rec_obj = mpatches.Rectangle((x_bottom_left, y_bottom_left), 
            ##                              2 * offset, 2 * offset, 
            x = int(x)
            y = int(y)
            rec_obj = mpatches.Circle((x, y), offset,
                                         linewidth = self.linewidth, 
                                         edgecolor = 'cyan', 
                                         facecolor='none')
            ax_img.add_patch(rec_obj)


    def plot_peaks(self):
        peak_list = self.peak_list
        offset = 4

        ax_raw = self.ax_list[0]
        for x, y in peak_list:
            x = int(x)
            y = int(y)
            rec_obj = mpatches.Circle((x, y), offset // 2,
                                      linewidth = self.linewidth, 
                                      edgecolor = 'magenta', 
                                      facecolor='none')
            ax_raw.add_patch(rec_obj)

        ax_img = self.ax_list[1]
        x_min, y_min, x_max, y_max = self.cheetah_geom_dict[self.panel]
        for x, y in peak_list:
            x = int(x) - x_min
            y = int(y) - y_min
            rec_obj = mpatches.Circle((x, y), offset // 2,
                                      linewidth = self.linewidth, 
                                      edgecolor = 'magenta', 
                                      facecolor='none')
            ax_img.add_patch(rec_obj)


        # BAD CODING PRACTICE, not good for maintainence, change it later!!!
        ax_img = self.ax_list[3]

        img  = self.img
        mask = self.mask_predicted

        size_img_tuple  = img.shape[-2:]
        size_crop_tuple = mask.shape[-2:]

        x_min, y_min, x_max, y_max = self.cheetah_geom_dict[self.panel]
        for x, y in peak_list:
            x = int(x) - x_min
            y = int(y) - y_min

            # Transform...
            coord_tuple = (y, x)
            y, x = coord_img_to_crop(coord_tuple, size_img_tuple, size_crop_tuple)

            rec_obj = mpatches.Circle((x, y), offset // 2,
                                      linewidth = self.linewidth, 
                                      edgecolor = 'magenta', 
                                      facecolor='none')
            ax_img.add_patch(rec_obj)


    def plot_mask(self):
        mask = self.mask
        size_y, size_x = mask.shape[-2:]

        ax_img = self.ax_list[2]
        im = ax_img.imshow(mask, vmin = 0, vmax = 1)
        im.set_cmap('gray')
        ax_img.invert_yaxis()

        b_offset = self.b_offset
        y_bmin, x_bmin = 0, 0
        y_bmax, x_bmax = size_y, size_x
        ax_img.set_xlim([x_bmin - b_offset, x_bmax + b_offset])
        ax_img.set_ylim([y_bmin - b_offset, y_bmax + b_offset])



    ## def plot_mask_predicted(self):
    ##     mask = self.mask_predicted
    ##     size_y, size_x = mask.shape[-2:]

    ##     ax_img = self.ax_list[3]
    ##     im = ax_img.imshow(mask, vmin = 0, vmax = 1)
    ##     im.set_cmap('gray')
    ##     ax_img.invert_yaxis()

    ##     b_offset = self.b_offset
    ##     y_bmin, x_bmin = 0, 0
    ##     y_bmax, x_bmax = size_y, size_x
    ##     ax_img.set_xlim([x_bmin - b_offset, x_bmax + b_offset])
    ##     ax_img.set_ylim([y_bmin - b_offset, y_bmax + b_offset])



    def plot_mask_predicted_overlay(self):
        img  = self.img_true
        mask = self.mask_predicted
        size_y, size_x = mask.shape[-2:]

        ax_img = self.ax_list[3]
        im = ax_img.imshow(img, vmin = 0, vmax = self.threshold)
        im.set_cmap('gray')
        ax_img.invert_yaxis()

        ax_img = self.ax_list[3]
        im = ax_img.imshow(mask, vmin = 0, vmax = 1, alpha = 0.8)
        ## cmap1 = mcolors.ListedColormap(['none', '#06ff00'])
        cmap1 = mcolors.ListedColormap(['none', 'red'])
        ## cmap1 = mcolors.ListedColormap(['black', 'none'])
        ## cmap1 = mcolors.ListedColormap(['black', 'white'])
        ## im.set_cmap('Greens')
        im.set_cmap(cmap1)
        ax_img.invert_yaxis()

        b_offset = self.b_offset
        y_bmin, x_bmin = 0, 0
        y_bmax, x_bmax = size_y, size_x
        ax_img.set_xlim([x_bmin - b_offset, x_bmax + b_offset])
        ax_img.set_ylim([y_bmin - b_offset, y_bmax + b_offset])



    def plot_mask_predicted(self):
        ## mask = self.mask_true
        mask = self.mask_predicted
        size_y, size_x = mask.shape[-2:]

        ax_img = self.ax_list[4]
        im = ax_img.imshow(mask, vmin = 0, vmax = 1)
        im.set_cmap('gray')
        ax_img.invert_yaxis()

        b_offset = self.b_offset
        y_bmin, x_bmin = 0, 0
        y_bmax, x_bmax = size_y, size_x
        ax_img.set_xlim([x_bmin - b_offset, x_bmax + b_offset])
        ax_img.set_ylim([y_bmin - b_offset, y_bmax + b_offset])



    ## def plot_mask_true(self):
    ##     mask = self.mask_true
    ##     size_y, size_x = mask.shape[-2:]

    ##     ax_img = self.ax_list[4]
    ##     im = ax_img.imshow(mask, vmin = 0, vmax = 1)
    ##     im.set_cmap('gray')
    ##     ax_img.invert_yaxis()

    ##     b_offset = self.b_offset
    ##     y_bmin, x_bmin = 0, 0
    ##     y_bmax, x_bmax = size_y, size_x
    ##     ax_img.set_xlim([x_bmin - b_offset, x_bmax + b_offset])
    ##     ax_img.set_ylim([y_bmin - b_offset, y_bmax + b_offset])


    def adjust_margin(self):
        self.fig.subplots_adjust(
            ## top=1-0.049,
            ## bottom=0.049,
            left=0.042,
            right=1-0.042,
            hspace=0.2,
            wspace=0.2
        )


    def show(self, idx, linewidth = 1.0, filename = None):
        self.fig, self.ax_list = self.create_panels()

        self.linewidth = linewidth

        self.fetch_data(idx)

        self.plot_geom()
        self.plot_raw_img()
        self.plot_img()
        self.plot_found()
        self.plot_indexed()
        self.plot_peaks()
        self.plot_mask()
        self.plot_mask_predicted_overlay()
        self.plot_mask_predicted()
        ## self.plot_mask_true()

        ## plt.tight_layout()
        self.adjust_margin()

        title = f"Loss: {self.loss:12.6f}"
        plt.suptitle(title, y = 0.95)
        if not isinstance(filename, str): 
            plt.show()
            plt.close()
        else:
            # Set up drc...
            DRCPDF         = "pdfs"
            drc_cwd        = os.getcwd()
            prefixpath_pdf = os.path.join(drc_cwd, DRCPDF)
            if not os.path.exists(prefixpath_pdf): os.makedirs(prefixpath_pdf)

            # Specify file...
            fl_pdf = f"{filename}.pdf"
            path_pdf = os.path.join(prefixpath_pdf, fl_pdf)

            # Export...
            plt.savefig(path_pdf, dpi = 600)
            plt.close()




## timestamp = None
## timestamp = "2022_0826_2112_07"
## timestamp = "2022_0826_2121_54"
## timestamp = "2022_0826_2142_15"
## timestamp = "2022_0826_2151_39"
## timestamp = "2022_0826_2204_12"
## timestamp = "2022_0826_2205_23"
timestamp = "2022_0826_2359_14"

# Set up parameters for an experiment...
fl_csv               = 'datasets.csv'
drc_project          = os.getcwd()
size_sample_train    = 200
size_sample_validate = 200
frac_train           = 0.005
frac_validate        = None
dataset_usage        = 'train'

size_batch = 1
lr         = 1e-4
seed       = 0

# [[[ DATASET ]]]
# Config the dataset...
config_dataset = ConfigDataset( fl_csv         = fl_csv,
                                drc_project    = drc_project,
                                size_sample    = size_sample_train, 
                                dataset_usage  = dataset_usage,
                                trans          = None,
                                frac_train     = frac_train,
                                frac_validate  = frac_validate,
                                mpi_comm       = mpi_comm,
                                seed           = seed, 
                                add_channel_ok = True, )

# Define the training set
dataset = SFXPanelDatasetMini(config_dataset)
dataset.mpi_cache_img()

if mpi_rank == 0: 
    MPI.Finalize()

    method = UNet(in_channels = 1, out_channels = 1)

    # [[[ MODEL ]]]
    # Config the model...
    pos_pixel = sum([ dataset[i][-1].sum() for i in range(len(dataset)) ])
    neg_pixel = len(dataset) * dataset[0][0].size - pos_pixel
    pos_weight = neg_pixel / pos_pixel
    config_peakfinder = ConfigPeakFinderModel( method     = method,
                                               pos_weight = pos_weight, )
    model = PeakFinderModel(config_peakfinder)

    # Initialize weights...
    def init_weights(module):
        if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
            module.weight.data.normal_(mean = 0.0, std = 0.02)
    model.apply(init_weights)


    # [[[ CHECKPOINT ]]]
    drc_cwd          = os.getcwd()
    DRCCHKPT         = "chkpts"
    prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
    if not os.path.exists(prefixpath_chkpt): os.makedirs(prefixpath_chkpt)
    path_chkpt = None
    if timestamp is not None:
        fl_chkpt   = f"{timestamp}.train.chkpt"
        path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)

    # Let's plot the first image...
    ## idx = 3
    ## idx = 75
    disp_manager = SanityChecker( model      = model,
                                  dataset    = dataset, 
                                  path_chkpt = path_chkpt, 
                                  figsize    = (32,15) )
    ## for idx in (3, 6, 19, 38):
    for idx in random.sample(range(size_sample_train), k = 20):
        print(idx)
        fl_pdf = f'pf.trained.idx{idx:06d}'
        disp_manager.show(idx)
        disp_manager.show(idx, linewidth = 0.1, filename = fl_pdf)

    ## idx = 56
    ## idx = 159
    ## ## fl_pdf = 'pf.init'
    ## fl_pdf = f'pf.trained.idx{idx:06d}'
    ## disp_manager.show(idx)
    ## disp_manager.show(idx, linewidth = 0.1, filename = fl_pdf)
