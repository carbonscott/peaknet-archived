#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
from stream_parser import ConfigParam, StreamParser, GeomInterpreter
import matplotlib              as mpl
import matplotlib.pyplot       as plt
import matplotlib.colors       as mcolors
import matplotlib.patches      as mpatches
import matplotlib.transforms   as mtransforms
import matplotlib.font_manager as font_manager

class VizCheetahGeom:
    def __init__(self, data, title, figsize, **kwargs):
        self.data    = data
        self.title   = title
        self.figsize = figsize
        for k, v in kwargs.items(): setattr(self, k, v)

        ## self.config_fonts()
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
        plt.rcParams['font.size']   = 18

        return None


    def create_panels(self):
        ncols = 1
        nrows = 1

        fig   = plt.figure(figsize = self.figsize)
        gspec = fig.add_gridspec( nrows, ncols,
                                  ## width_ratios  = [1, 4/20, 4/20, 4/20, 4/20, 4/20, 1/20],·
                                  ## height_ratios = [4/20, 4/20, 4/20, 4/20, 4/20], 
                                )
        ax_list = [ fig.add_subplot(gspec[i, j], aspect = 1) for i in range(nrows) for j in range(ncols) ]

        self.ncols = ncols
        self.nrows = nrows

        return fig, ax_list


    def config_colorbar(self, vmin = -1, vcenter = 0, vmax = 1):
        # Plot image...
        self.divnorm = mcolors.TwoSlopeNorm(vcenter = vcenter, vmin = vmin, vmax = vmax)


    def plot(self):
        ax_img  = self.ax_list[0]
        ## ax_img.set_xticks([])
        ## ax_img.set_yticks([])

        data = self.data

        y_bmin, x_bmin, y_bmax, x_bmax = 0, 0, 0, 0
        for k, (x_min, y_min, x_max, y_max) in data.items():
            w = x_max - x_min
            h = y_max - y_min
            rec = mpatches.Rectangle(xy = (x_min, y_min), width = w, height = h, facecolor='none', edgecolor = 'green')

            ax_img.add_patch(rec)

            y_text = (y_min + y_max) / 2
            x_text = (x_min + x_max) / 2
            ax_img.text(x = x_text, y = y_text, s = k, 
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

        self.plot()

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
            plt.savefig(path_pdf, dpi = 100)


## path_stream = '/reg/data/ana03/scratch/cwang31/pf/streams/cxic0415_0101.stream'
## config_stream_parser = ConfigParam( path_stream = path_stream )
## stream_parser = StreamParser(config_stream_parser)
## stream_parser.parse()
## match_dict = stream_parser.match_dict
## fl_match_dict = 'match_dict.pickle'
## with open(fl_match_dict, 'wb') as fh:
##     pickle.dump(match_dict, fh, protocol = pickle.HIGHEST_PROTOCOL)

fl_match_dict = 'match_dict.pickle'
with open(fl_match_dict, 'rb') as fh:
    match_dict = pickle.load(fh)
geom_dict = match_dict['geom']
cheetah_geom_dict = GeomInterpreter(geom_dict).interpret()

disp_manager = VizCheetahGeom(data = cheetah_geom_dict, title = '', figsize = (12,8))
disp_manager.show()
