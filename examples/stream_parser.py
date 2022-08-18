#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import logging
import pickle

logger = logging.getLogger(__name__)

class ConfigParam:

    def __init__(self, name_cls = '', **kwargs):
        self.name_cls = name_cls

        self.kwargs = kwargs
        for k, v in kwargs.items(): setattr(self, k, v)


    def report(self):
        logger.info(f"___/ Configure '{self.name_cls}' \___")
        for k, v in self.__dict__.items():
            if k == 'kwargs': continue
            logger.info(f"KV - {k:16s} : {v}")




class StreamParser:
    '''
    Extract successfully indexed peaks previously found by psocake peak finder.

    Remark: Avoid parsing a stream file if you can.  CrystFEL stream file has
    several levels of ambiguities to handle (:, =, 2 kinds of tabulated data).
    Python's lack of non-backtracking in regex (the '(?>)' block) will
    exacerbate the file parsing.

    '''

    def __init__(self, config):
        self.path_stream = getattr(config, 'path_stream', None)

        self.marker_dict = {
            "CHUNK_START"      : "----- Begin chunk -----",
            "CHUNK_END"        : "----- End chunk -----",
            "GEOM_START"       : "----- Begin geometry file -----",
            "GEOM_END"         : "----- End geometry file -----",
            "PEAK_LIST_START"  : "Peaks from peak search",
            "PEAK_LIST_END"    : "End of peak list",
            "CRYSTAL_START"    : "--- Begin crystal",
            "CRYSTAL_END"      : "--- End crystal",
            "REFLECTION_START" : "Reflections measured after indexing",
            "REFLECTION_END"   : "End of reflections",
        }

        self.geom_dict = { 'dim' : {}, 'panel' : {} }
        self.peak_dict = {}


    def parse(self):
        '''
        Return a dictionary of events that contain successfully indexed peaks
        found by psocake peak finder.
        '''
        # Define state variable for parse/skipping lines...
        is_chunk_found = False
        save_peak_ok   = False
        in_peaklist    = False
        in_indexedlist = False
        in_geom        = False

        # Import marker...
        marker_dict = self.marker_dict

        # Start parsing a stream file by chunks...
        path_stream = self.path_stream
        with open(path_stream,'r') as fh:
            for line in fh:
                line = line.strip()

                # To find a new chunk...
                # Consider chunk is found...
                if line == marker_dict["CHUNK_START"]: 
                    is_chunk_found = True
                    save_peak_ok   = False    # ...Don't save any peaks by default

                # Consider chunk not found at the end of a chunk...
                if line == marker_dict["CHUNK_END"]: is_chunk_found = False

                # Skip parsing statements below if a chunk is not found...
                if not is_chunk_found: continue

                # To decide whether to save peaks...
                # Look up filename...
                if line.startswith("Image filename: "):
                    filename = line[line.rfind(':') + 1:] # e.g. 'Image filename: /xxx/cxig3514_0041.cxi'
                    filename = filename.strip()

                # Look up event number...
                if line.startswith("Event: "):
                    event_num = line[line.rfind('/') + 1:] # e.g. 'Event: //17'
                    event_num = int(event_num)

                # Look up indexing status...
                if line.startswith("indexed_by"):
                    status_indexing = line[line.rfind('=') + 1:].strip() # e.g. 'indexed_by = none'

                    # Allow peak saving when indexing is successful...
                    if status_indexing != 'none': save_peak_ok = True

                # Don't save any peaks if indexing is not successful...
                if not save_peak_ok: continue

                # To save results from events in each file...
                # Save by filename
                if not filename in self.peak_dict: self.peak_dict[filename] = {}

                # Save by event number
                if not event_num in self.peak_dict[filename]:
                    self.peak_dict[filename][event_num] = {}

                # Find a peak list...
                if line == marker_dict["PEAK_LIST_START"]:
                    in_peaklist = True
                    is_first_line_in_peaklist = True
                    continue

                # Exit a peak list...
                if line == marker_dict["PEAK_LIST_END"]:
                    in_peaklist = False
                    continue

                # To Save peaks in a peak list...
                if in_peaklist:
                    # Skip the header...
                    if is_first_line_in_peaklist:
                        is_first_line_in_peaklist = False
                        continue

                    # Saving...
                    dim1, dim2, _, _, panel = line.split()

                    if not panel in self.peak_dict[filename][event_num]: 
                        self.peak_dict[filename ] \
                                      [event_num] \
                                      [panel    ] = { 'found' : [], 'indexed' : [] }

                    self.peak_dict[filename ] \
                                  [event_num] \
                                  [panel    ] \
                                  ['found'  ].append((float(dim1), 
                                                      float(dim2),))
                    continue

                # Find an indexed list...
                if line == marker_dict["REFLECTION_START"]:
                    in_indexedlist = True
                    is_first_line_in_indexedlist = True
                    continue

                # Exit an indexed list...
                if line == marker_dict["REFLECTION_END"]:
                    in_indexedlist = False
                    continue

                # To Save peaks in a indexed list...
                if in_indexedlist:
                    # Skip the header...
                    if is_first_line_in_indexedlist:
                        is_first_line_in_indexedlist = False
                        continue

                    # Saving...
                    _, _, _, _, _, _, _, dim1, dim2, panel = line.split()

                    # If the panel doesn't have found peak, skip it...
                    if not panel in self.peak_dict[filename][event_num]: continue

                    # Otherwise, save it...
                    self.peak_dict[filename ] \
                                  [event_num] \
                                  [panel    ] \
                                  ['indexed'].append((float(dim1), 
                                                      float(dim2),))
                    continue




# [[[ EXAMPLE ]]]

path_stream = '/reg/data/ana15/cxi/cxig3514/scratch/cwang31/psocake/r0041/cxig3514_0041.stream'
## path_stream = '/reg/data/ana15/cxi/cxig3514/scratch/cwang31/psocake/r0041/cxig3514_0041_d100.stream'
## path_stream = '/reg/data/ana15/cxi/cxig3514/scratch/cwang31/psocake/r0041/test2.stream'
## path_stream = '/reg/data/ana15/cxi/cxig3514/scratch/cwang31/psocake/r0041/test4.stream'
## path_stream = '/reg/data/ana15/cxi/cxig3514/scratch/cwang31/psocake/r0041/test.stream'
## path_stream = '/reg/data/ana03/scratch/cwang31/pf/streams/test.stream'
config_stream_parser = ConfigParam( path_stream = path_stream )
stream_parser = StreamParser(config_stream_parser)
stream_parser.parse()
peak_dict = stream_parser.peak_dict

## fl_peak_dict = 'peak_dict.pickle'
## with open(fl_peak_dict, 'wb') as fh:
##     pickle.dump(peak_dict, fh, protocol = pickle.HIGHEST_PROTOCOL)
