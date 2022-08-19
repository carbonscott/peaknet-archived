#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import h5py
import random
import numpy as np
from scipy.spatial import cKDTree
from stream_parser import ConfigParam, StreamParser, GeomInterpreter
import matplotlib              as mpl
import matplotlib.pyplot       as plt
import matplotlib.colors       as mcolors
import matplotlib.patches      as mpatches
import matplotlib.transforms   as mtransforms
import matplotlib.font_manager as font_manager

seed = 0
random.seed(seed)

class VizCheetahGeom:
    def __init__(self, geom_dict, img, peak_dict, indexed_dict, title, figsize, **kwargs):
        self.geom_dict = geom_dict
        self.img       = img
        self.peak_dict = peak_dict
        self.indexed_dict = indexed_dict
        self.title     = title
        self.figsize   = figsize
        for k, v in kwargs.items(): setattr(self, k, v)

        self.config_fonts()
        ## self.config_colorbar()

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
        plt.rcParams['font.size']   = 12

        return None


    def create_panels(self):
        ncols = 2
        nrows = 1

        fig   = plt.figure(figsize = self.figsize)
        gspec = fig.add_gridspec( nrows, ncols,
                                  width_ratios  = [1, 1/20],
                                  ## height_ratios = [4/20, 4/20, 4/20, 4/20, 4/20], 
                                )
        ax_list = [ fig.add_subplot(gspec[i, j], aspect = 1) for i in range(nrows) for j in range(ncols) ]

        self.ncols = ncols
        self.nrows = nrows

        return fig, ax_list


    def config_colorbar(self, vmin = -1, vcenter = 0, vmax = 1):
        # Plot image...
        self.divnorm = mcolors.TwoSlopeNorm(vcenter = vcenter, vmin = vmin, vmax = vmax)


    def plot_geom(self):
        ax_img  = self.ax_list[0]
        ## ax_img.set_xticks([])
        ## ax_img.set_yticks([])

        geom_dict = self.geom_dict

        y_bmin, x_bmin, y_bmax, x_bmax = 0, 0, 0, 0
        for k, (x_min, y_min, x_max, y_max) in geom_dict.items():
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


    def plot_img(self):
        img = self.img
        ax_img  = self.ax_list[0]
        ax_cbar = self.ax_list[1]
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


    def plot_peaks(self):
        peak_dict = self.peak_dict
        ax_img = self.ax_list[0]
        offset = 4
        for panel, peak_list in peak_dict.items():
            for x, y in peak_list:
                x_bottom_left = int(x - offset)
                y_bottom_left = int(y - offset)

                rec_obj = mpatches.Rectangle((x_bottom_left, y_bottom_left), 
                                             2 * offset, 2 * offset, 
                                             linewidth = 0.2, 
                                             edgecolor = 'yellow', 
                                             facecolor='none')
                ax_img.add_patch(rec_obj)


    def plot_indexed(self):
        indexed_dict = self.indexed_dict
        ax_img = self.ax_list[0]
        offset = 4
        for panel, indexed_list in indexed_dict.items():
            for x, y in indexed_list:
                x_bottom_left = int(x - offset)
                y_bottom_left = int(y - offset)

                ## rec_obj = mpatches.Rectangle((x_bottom_left, y_bottom_left), 
                ##                              2 * offset, 2 * offset, 
                x = int(x)
                y = int(y)
                rec_obj = mpatches.Circle((x, y), offset,
                                             linewidth = 0.2, 
                                             edgecolor = 'cyan', 
                                             facecolor='none')
                ax_img.add_patch(rec_obj)


    def plot_indexed_peaks(self):
        peak_dict    = self.peak_dict
        indexed_dict = self.indexed_dict

        ax_img = self.ax_list[0]
        offset = 4
        for panel, peak_list in peak_dict.items():
            if not panel in indexed_dict: continue

            indexed_list = indexed_dict[panel]

            peak_list_tree = cKDTree(peak_list)
            indexed_list_tree = cKDTree(indexed_list)

            idx_indexed_peak_list = peak_list_tree.query_ball_tree(indexed_list_tree, r = 5)

            for idx, neighbor_list in enumerate(idx_indexed_peak_list):
                if len(neighbor_list) == 0: continue

                x, y = peak_list[idx]
                x = int(x)
                y = int(y)
                rec_obj = mpatches.Circle((x, y), offset // 2,
                                          linewidth = 0.2, 
                                          edgecolor = 'magenta', 
                                          facecolor='none')
                ax_img.add_patch(rec_obj)




    def adjust_margin(self):
        self.fig.subplots_adjust(
            top=1-0.049,
            bottom=0.049,
            left=0.042,
            right=1-0.042,
            hspace=0.2,
            wspace=0.2
        )


    def show(self, filename = None):
        self.fig, self.ax_list = self.create_panels()

        self.plot_geom()
        self.plot_img()
        self.plot_peaks()
        self.plot_indexed()
        self.plot_indexed_peaks()

        ## plt.tight_layout()
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




# Get geom from stream...
fl_match_dict = 'match_dict.pickle'
with open(fl_match_dict, 'rb') as fh:
    match_dict = pickle.load(fh)
geom_dict = match_dict['geom']
cheetah_geom_dict = GeomInterpreter(geom_dict).interpret()


# Get image from cxi...
fl_cxi = list(match_dict['chunk'].keys())[0]
event_crystfel_list = list(match_dict['chunk'][fl_cxi].keys())
event_crystfel = random.choice(event_crystfel_list)
with h5py.File(fl_cxi, 'r') as fh:
    event_lcls_list = fh["LCLS/eventNumber"][()]

    event_lcls = event_lcls_list[event_crystfel]

    img = fh["/entry_1/instrument_1/detector_1/data"][event_crystfel]

# Get the right chunk...
chunk_dict = match_dict['chunk'][fl_cxi][event_crystfel]

# Find all peaks to highlight...
peak_dict = {}
indexed_dict = {}
for panel, peak_saved_dict in chunk_dict.items():
    peak_dict[panel] = peak_saved_dict['found']
    indexed_dict[panel] = peak_saved_dict['indexed']

disp_manager = VizCheetahGeom(geom_dict = cheetah_geom_dict, 
                              img = img, 
                              peak_dict = peak_dict, 
                              indexed_dict = indexed_dict, 
                              title = '', 
                              figsize = (12,10))
disp_manager.show()

fl_pdf = 'viz_panel'
disp_manager.show(filename = fl_pdf)
