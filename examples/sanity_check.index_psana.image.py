#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np

from scipy import ndimage

from peaknet.datasets.utils           import PsanaImg
from peaknet.datasets.inference_psana import ConfigDataset, SFXRandomSubset
from peaknet.methods.unet             import UNet
from peaknet.model                    import ConfigPeakFinderModel, PeakFinderModel
from peaknet.datasets.stream_parser   import GeomInterpreter
from peaknet.datasets.transform       import center_crop, coord_img_to_crop

import matplotlib              as mpl
import matplotlib.pyplot       as plt
import matplotlib.colors       as mcolors
import matplotlib.patches      as mpatches
import matplotlib.transforms   as mtransforms
import matplotlib.font_manager as font_manager

class SanityChecker:
    def __init__(self, model, img, path_chkpt, figsize, **kwargs):
        self.model      = model
        self.img        = img
        self.path_chkpt = path_chkpt
        self.figsize    = figsize

        for k, v in kwargs.items(): setattr(self, k, v)

        # Load model to gpus if available...
        self.device = 'cpu'
        if self.path_chkpt is not None and torch.cuda.is_available():
            self.device = torch.cuda.current_device()

            chkpt = torch.load(self.path_chkpt)
            self.model.load_state_dict(chkpt)
            self.model = torch.nn.DataParallel(self.model.method).to(self.device)

        self.config_fonts()
        ## self.config_colorbar()

        self.linewidth = 1.0

        self.b_offset = 2
        self.Sigmoid = torch.nn.Sigmoid()
        self.sigmoid_threshold = 0.9

        return None


    def fetch_data(self):
        # Load model and testing configuration...
        model = self.model

        # Load model state...
        model.eval()

        img = self.img
        img = img[None, ]

        # Add a fake batch dim and move the data to gpu if available...
        # The model is trained with the extra dimension
        batch_img  = torch.Tensor(img [None,])
        batch_img  = batch_img.to(self.device)

        # Find the predicted mask...
        with torch.no_grad():
            batch_mask_predicted = model.forward(batch_img)

        ## batch_mask_predicted[  batch_mask_predicted < 100 ] = 0
        ## batch_mask_predicted[~(batch_mask_predicted < 100)] = 1

        # Crop the original image...
        size_y, size_x = batch_mask_predicted.shape[-2:]
        batch_img_true, offset_tuple = center_crop(batch_img, size_y, size_x, return_offset_ok = True)
        batch_img_true = batch_img_true.cpu().detach().numpy()
        img_true = batch_img_true.reshape(*batch_img_true.shape[-3:])    # Remove fake batch layer

        # Save them to cpu...
        batch_mask_predicted = batch_mask_predicted.cpu().detach().numpy()
        mask_predicted       = batch_mask_predicted.reshape(*batch_mask_predicted.shape[-3:])    # Remove fake batch layer

        img            = img           [0]
        mask_predicted = mask_predicted[0]
        img_true       = img_true      [0]

        # Sync for visualization...
        self.img               = img
        self.mask_predicted    = mask_predicted
        self.img_true          = img_true
        self.offset_tuple      = offset_tuple


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
        ncols = 7
        nrows = 2

        fig   = plt.figure(figsize = self.figsize)
        gspec = fig.add_gridspec( nrows, ncols,
                                  width_ratios  = [1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/20],
                                  height_ratios = [1/2, 1/2], 
                                )
        ax_list = [ fig.add_subplot(gspec[0:2, 0:2], aspect = 1) ]
        ax_list.append( fig.add_subplot(gspec[0:2, 2:4], aspect = 1) )
        ax_list.append( fig.add_subplot(gspec[0:2, 4:6], aspect = 1) )
        ax_list.append( fig.add_subplot(gspec[:, 6], aspect = 1) )

        self.ncols = ncols
        self.nrows = nrows

        return fig, ax_list


    def config_colorbar(self, vmin = -1, vcenter = 0, vmax = 1):
        # Plot image...
        self.divnorm = mcolors.TwoSlopeNorm(vcenter = vcenter, vmin = vmin, vmax = vmax)


    def plot_img(self):
        img = self.img_true

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
        vmax = self.threshold

        im = ax_img.imshow(img, vmin = vmin, vmax = vmax)
        im.set_cmap('gray')
        plt.colorbar(im, cax = ax_cbar, orientation="vertical", pad = 0.05)

        ax_img.invert_yaxis()

        size_y, size_x = img.shape[-2:]
        b_offset = self.b_offset
        y_bmin, x_bmin = 0, 0
        y_bmax, x_bmax = size_y, size_x
        ax_img.set_xlim([x_bmin - b_offset, x_bmax + b_offset])
        ax_img.set_ylim([y_bmin - b_offset, y_bmax + b_offset])


        return None


    def plot_mask_predicted_overlay(self):
        img  = self.img_true
        mask = self.mask_predicted
        size_y, size_x = mask.shape[-2:]

        ax_img = self.ax_list[1]
        im = ax_img.imshow(img, vmin = 0, vmax = self.threshold)
        im.set_cmap('gray')
        ax_img.invert_yaxis()

        # Threshold
        adu_threshold = self.adu_threshold
        mask[mask < adu_threshold] = 0.0

        ax_img = self.ax_list[1]
        im = ax_img.imshow(mask, vmin = 0, vmax = 1, alpha = 1.0)
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

        # Put box on peaks...
        peak_labeled, num_peak = ndimage.label(mask)
        peak_pos_list = ndimage.center_of_mass(mask, peak_labeled, range(num_peak))

        # Add box
        offset = 3
        for y, x in peak_pos_list:
            x_bottom_left = x - offset
            y_bottom_left = y - offset

            rec_obj = mpatches.Rectangle((x_bottom_left, y_bottom_left), 
                                         2 * offset, 2 * offset, 
                                         linewidth = self.linewidth, 
                                         edgecolor = 'yellow', 
                                         facecolor='none')
            ax_img.add_patch(rec_obj)



    def plot_mask_predicted(self):
        mask = self.mask_predicted
        size_y, size_x = mask.shape[-2:]

        # Threshold
        adu_threshold = self.adu_threshold
        mask[mask < adu_threshold] = 0.0

        ax_img = self.ax_list[2]
        im = ax_img.imshow(mask, vmin = 0, vmax = 1)
        im.set_cmap('gray')
        ax_img.invert_yaxis()

        b_offset = self.b_offset
        y_bmin, x_bmin = 0, 0
        y_bmax, x_bmax = size_y, size_x
        ax_img.set_xlim([x_bmin - b_offset, x_bmax + b_offset])
        ax_img.set_ylim([y_bmin - b_offset, y_bmax + b_offset])



    def adjust_margin(self):
        self.fig.subplots_adjust(
            ## top=1-0.049,
            ## bottom=0.049,
            left=0.042,
            right=1-0.042,
            hspace=0.2,
            wspace=0.2
        )


    def show(self, linewidth = 1.0, filename = None):
        self.fig, self.ax_list = self.create_panels()

        self.linewidth = linewidth

        self.fetch_data()

        self.plot_img()
        self.plot_mask_predicted_overlay()
        self.plot_mask_predicted()

        ## plt.tight_layout()
        self.adjust_margin()

        ## title = f"Loss: {self.loss:12.6f}"
        ## plt.suptitle(title, y = 0.95)
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
## timestamp = "2022_0907_2327_35"    # frac_train = 0.5, pos_weight = 1  , suprisingly good
## timestamp = "2022_0907_2328_34"    # frac_train = 0.5, pos_weight = 0.1, less likely to predict peaks
## timestamp = "2022_0907_2328_45"    # frac_train = 0.5, pos_weight = 10, probably too aggresive to predict peaks
## timestamp = "2022_0908_1011_40"    # frac_train = 0.5, pos_weight = 5, pretty good.
timestamp = "2022_0908_1013_52"    # frac_train = 0.5, pos_weight = 2, pretty good.
## timestamp = "2022_0908_1032_37"    # frac_train = 0.5, pos_weight = 2.5,

## exp           = 'mfxlv4920'
## run           = 131
## mode          = 'idx'
## detector_name = 'MfxEndstation.0:Epix10ka2M.0'

exp           = 'cxic0415'
run           = 85
mode          = 'idx'
detector_name = 'CxiDs1.0:Cspad.0'
psana_img = PsanaImg(exp, run, mode, detector_name)

## event = 142027
event = 18537
## event = 23615
## img = psana_img.get(event, 1, 'calib')
img = psana_img.get(event, None, 'image')
## img = img[200:1600, 200:1600]
## img[img < 400] = 0


# [[[ MODEL ]]]
# Config the model...
method = UNet(in_channels = 1, out_channels = 1)
pos_weight = 2.0
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
disp_manager = SanityChecker( model         = model,
                              img           = img, 
                              path_chkpt    = path_chkpt, 
                              adu_threshold = 100,
                              figsize       = (18,6) )

fl_pdf = f'pf.inference.{exp}.{run}.{event:06d}'
disp_manager.show()
## disp_manager.show(linewidth = 1.0, filename = fl_pdf)


## plt.hist(disp_manager.mask_predicted.reshape(-1), 150)
## plt.show()
