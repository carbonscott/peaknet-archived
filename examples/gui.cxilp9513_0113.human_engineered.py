#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from pyqtgraph.Qt import QtWidgets

from img_labeler.layout import MainLayout
from img_labeler.window import Window
from img_labeler.data   import FastData

import socket

def run(config_data):
    # Main event loop
    app = QtWidgets.QApplication([])

    # Layout
    layout = MainLayout()

    # Data
    data_manager = FastData(config_data)

    # Window
    win = Window(layout, data_manager)
    win.config()
    win.show()

    sys.exit(app.exec_())


class ConfigData:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items(): setattr(self, k, v)

config_data = ConfigData( path_fastdata = "/reg/data/ana03/scratch/cwang31/pf/fastdata/cxilp9515.human_engineered.fastdata",
                          username = os.environ.get('USER'),
                          seed     = 0, )

run(config_data)
