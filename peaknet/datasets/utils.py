#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import psana

class PsanaImg:
    """
    It serves as an image accessing layer based on the data management system
    psana in LCLS.  
    """

    def __init__(self, exp, run, mode, detector_name):
        # Biolerplate code to access an image
        # Set up data source
        self.datasource_id = f"exp={exp}:run={run}:{mode}"
        self.datasource    = psana.DataSource( self.datasource_id )
        self.run_current   = next(self.datasource.runs())
        self.timestamps    = self.run_current.times()

        # Set up detector
        self.detector = psana.Detector(detector_name)


    def __len__(self):
        return len(self.timestamps)


    def get(self, event_num, id_panel = None, mode = "calib"):
        # Fetch the timestamp according to event number...
        timestamp = self.timestamps[event_num]

        # Access each event based on timestamp...
        event = self.run_current.event(timestamp)

        # Only two modes are supported...
        assert mode in ("raw", "calib", "image"), \
            f"Mode {mode} is not allowed!!!  Only 'raw', 'calib' and 'image' are supported."

        # Fetch image data based on timestamp from detector...
        read = { "raw"   : self.detector.raw,
                 "calib" : self.detector.calib,
                 "image" : self.detector.image,}
        data = read[mode](event)
        img  = data[int(id_panel)] if id_panel is not None else data

        return img
