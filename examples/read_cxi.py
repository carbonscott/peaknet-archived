#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import random

import matplotlib              as mpl
import matplotlib.pyplot       as plt
import matplotlib.colors       as mcolors
import matplotlib.patches      as mpatches
import matplotlib.transforms   as mtransforms
import matplotlib.font_manager as font_manager

seed = 0
random.seed(seed)

class Viz:

    def __init__(self, data, idx_viz_sample_list, peak_x, peak_y, offset, vmin, vmax, title,  figsize, **kwargs):
        self.data                 = data
        self.title                = title
        self.figsize              = figsize
        self.idx_viz_sample_list  = idx_viz_sample_list
        self.peak_x               = peak_x
        self.peak_y               = peak_y
        self.offset               = offset
        self.vmin                 = vmin
        self.vmax                 = vmax
        self.threshold            = 0.0
        for k, v in kwargs.items(): setattr(self, k, v)

        ## self.config_fonts()

        return None


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
        plt.rcParams['font.size']   = 18

        return None


    def create_panels(self):
        nrows, ncols = 5, 2 + 5
        fig = plt.figure(figsize = self.figsize)

        gspec   = fig.add_gridspec(nrows, ncols,
                                   width_ratios  = [1, 4/20, 4/20, 4/20, 4/20, 4/20, 1/20], 
                                   height_ratios  = [4/20, 4/20, 4/20, 4/20, 4/20], )

        ax_list = (fig.add_subplot(gspec[0:5,0], aspect = 1), )
        ax_list+= tuple(fig.add_subplot(gspec[i,1 + j], aspect = 1) for i in range(5) for j in range(5))
        ax_list+= (fig.add_subplot(gspec[0:5,-1],  aspect = 20), )

        return fig, ax_list


    def plot(self): 
        ax_img  = self.ax_list[0]
        ax_cbar = self.ax_list[-1]
        ## self.config_colorbar()
        ## im = ax_img.imshow(self.data, norm = self.divnorm)

        std_level = 4
        img_mean = np.mean(img)
        img_std  = np.std(img)
        threshold = img_mean + img_std * std_level
        ## img[img < threshold] = 0.0
        ## img[img >= threshold] = 1.0
        self.threshold = threshold
        vmin = 0
        vmax = threshold

        im = ax_img.imshow(self.data, vmin = vmin, vmax = vmax)
        im.set_cmap('gray')
        plt.colorbar(im, cax = ax_cbar, orientation="vertical", pad = 0.05)


    def plot_rectangle(self):
        ax_img = self.ax_list[0]
        peak_x = self.peak_x
        peak_y = self.peak_y
        offset = self.offset
        idx_viz_sample_list = self.idx_viz_sample_list
        for x, y in zip(peak_x, peak_y):
            x_bottom_left = int(x - offset)
            y_bottom_left = int(y - offset)

            rec_obj = mpatches.Rectangle((x_bottom_left, y_bottom_left), 
                                         2 * offset, 2 * offset, 
                                         linewidth = 1.0, 
                                         edgecolor = 'yellow', 
                                         facecolor='none')
            ax_img.add_patch(rec_obj)


        for i in idx_viz_sample_list:
            x, y = peak_x[i], peak_y[i]
            x_bottom_left = int(x - offset)
            y_bottom_left = int(y - offset)

            rec_obj = mpatches.Rectangle((x_bottom_left, y_bottom_left), 
                                         2 * offset, 2 * offset, 
                                         linewidth = 1.0, 
                                         edgecolor = 'green', 
                                         facecolor='none')
            ax_img.add_patch(rec_obj)


    def plot_patch(self):
        ax = self.ax_list[1:-1]
        size_y, size_x = self.data.shape
        peak_x = self.peak_x
        peak_y = self.peak_y
        offset = self.offset
        idx_viz_sample_list = self.idx_viz_sample_list
        for idx_patch, i in enumerate(idx_viz_sample_list):
            x, y = peak_x[i], peak_y[i]
            x_b = max(int(x - offset), 0)
            x_e = min(int(x + offset), size_x)
            y_b = max(int(y - offset), 0)
            y_e = min(int(y + offset), size_y)
            patch = self.data[y_b : y_e, x_b : x_e]

            std_level  = 0
            patch_mean = np.mean(patch)
            patch_std  = np.std (patch)
            threshold  = patch_mean + std_level * patch_std

            pos_pixel_selected_list = np.argwhere(~(patch < threshold))
            ## print(pos_pixel_selected_list)

            for pos_y_np, pos_x_np in pos_pixel_selected_list:
                pos_x_np = int(pos_x_np)
                pos_y_np = int(pos_y_np)

                rec_obj = mpatches.Circle((pos_x_np, pos_y_np),
                                          0.5,
                                          linewidth = 1.0, 
                                          edgecolor = 'red', 
                                          facecolor='none')
                ax[idx_patch].add_patch(rec_obj)


            ## patch[  patch < threshold ] = 0.0
            ## patch[~(patch < threshold)] = 1.0
            ## im = ax[idx_patch].imshow(patch, vmin = self.vmin, vmax = self.vmax)

            vmin = 0
            vmax = self.threshold
            im = ax[idx_patch].imshow(patch, vmin = 0, vmax = vmax)
            im.set_cmap('gray')
            ax[idx_patch].set_axis_off()


    def config_colorbar(self, vmin = -1, vcenter = 0, vmax = 1):
        # Plot image...
        self.divnorm = mcolors.TwoSlopeNorm(vcenter = vcenter, vmin = vmin, vmax = vmax)


    def adjust_margin(self):
        self.fig.subplots_adjust(
            top=0.981,
            bottom=0.049,
            left=0.042,
            right=0.981,
            hspace=0.2,
            wspace=0.2
        )


    def show(self, filename = None): 
        self.fig, self.ax_list = self.create_panels()

        self.plot()
        self.plot_rectangle()
        self.plot_patch()
        self.adjust_margin()

        ## plt.suptitle(self.title, y = 0.95)
        if not isinstance(filename, str): 
            plt.show()
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
            plt.savefig(path_pdf, dpi = 300)


drc      = '/reg/data/ana03/scratch/cwang31/pf/cxic0415/cwang31/psocake/r0101'
base_cxi = 'cxic0415_0101'
fl_cxi   = f'{base_cxi}.cxi'
path_cxi = os.path.join(drc, fl_cxi)

## data_cxi = h5py.File(path_cxi, "r")
with h5py.File(path_cxi, "r") as data_cxi:
    # ___/ HERE'S THE FORMAT \___
    idx_events = data_cxi["LCLS/eventNumber"            ]
    n_hits     = data_cxi["entry_1/result_1/nPeaks"     ]
    peak_x     = data_cxi["entry_1/result_1/peakXPosRaw"]
    peak_y     = data_cxi["entry_1/result_1/peakYPosRaw"]
    img_orig   = data_cxi["/entry_1/instrument_1/detector_1/data"]

    event_idx_dict = { event : idx for idx, event in enumerate(idx_events) }

    event = 17957
    idx   = event_idx_dict[event]
    img = img_orig[idx]

    size_sample = 25
    idx_viz_list = range(sum(peak_x[idx] > 0.0))
    idx_viz_sample_list = random.sample(idx_viz_list, size_sample)
    ## idx_viz_sample_list = idx_viz_list[:size_sample]

    # Only visualize top 25...
    offset = 4
    fl_pdf = f'labeled_peaks.{base_cxi}.sigma=0'
    disp_manager = Viz(data = img, idx_viz_sample_list = idx_viz_sample_list, peak_x = peak_x[idx], peak_y = peak_y[idx], offset = offset, vmin = 0, vmax = 1, title = "", figsize = (28, 10))
    disp_manager.show()
    disp_manager.show(filename = fl_pdf)
