#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import regex
import pickle

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

        self.regex_dict = self.init_regex()

        self.match_dict = {}


    def init_regex(self):
        regex_dict = {}

        # Parse peaks found...
        regex_dict['peak_found'] = regex.compile( 
            r"""
            (?x)
            \s*?              # Match whitespace or None
            (?:
                (?>
                    (?&FLOAT) # Match a floating number
                )
                \s+?          # Match whitespace at least once
            ){4}              # Match the whole construct 4 times

            (?&DET_PANEL)     # Match a detector panel

            (?(DEFINE)
                (?<FLOAT>
                    ([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)
                )
                (?<DET_PANEL>
                    (?: [0-9A-Za-z]+ )
                )
            )
            """
        )

        # Parse peaks found...
        regex_dict['peak_indexed'] = regex.compile( 
            r"""
            (?x)
            \s*?              # Match whitespace or None
            (?:
                (?>
                    (?&FLOAT) # Match a floating number
                )
                \s+?          # Match whitespace at least once
            ){9}              # Match the whole construct 4 times

            (?&DET_PANEL)     # Match a detector panel

            (?(DEFINE)
                (?<FLOAT>
                    ([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)
                )
                (?<DET_PANEL>
                    (?: [0-9A-Za-z]+ )
                )
            )
            """
        )

        # Parse colon item in event...
        regex_dict['event_colon_dict'] = regex.compile(
            r"""
            (?x)
            (?>
                (?P<EVENT_KEY>  # Match the keyword
                    (?:Image \s filename)
                |   (?:Event)
                |   (?:Image \s serial \s number)
                )
                :   # Match a colon
            )
            (?P<EVENT_VALUE>.+)  # Match the event number
            """
        )

        # Parse equal sign item in event...
        regex_dict['event_eq_dict'] = regex.compile(
            r"""
            (?x)
            (?>
                (?P<EVENT_KEY>  # Match the keyword
                    (?:hit                              )
                |   (?:indexed_by                       )
                |   (?:photon_energy_eV                 )
                |   (?:beam_divergence                  )
                |   (?:beam_bandwidth                   )
                |   (?:hdf5/LCLS/detector_1/EncoderValue)
                |   (?:hdf5/LCLS/photon_energy_eV       )
                |   (?:average_camera_length            )
                |   (?:num_peaks                        )
                |   (?:num_saturated_peaks              )
                |   (?:peak_resolution                  )
                )
                \s
                =   # Match a equal sign
                \s
            )
            (?P<EVENT_VALUE>.+)  # Match the event number
            """
        )

        # Parse detector gemoetry...
        regex_dict['geom'] = regex.compile(
            r"""
            (?x)
            (?>
                (?&DET_PANEL)
                /
                (?&COORD)
            )
            \s
            =
            \s
            (?&VALUE)

            (?(DEFINE)
                (?<DET_PANEL>
                    [0-9a-zA-Z]+?
                )
                (?<COORD>
                    (?:min_fs)
                |   (?:min_ss)
                |   (?:max_fs)
                |   (?:max_ss)
                )
                (?<VALUE> [0-9]+ )
            )
            """
        )

        return regex_dict


    def parse(self):
        '''
        Return a dictionary of matches that does ... .
        '''
        regex_dict = self.regex_dict

        # Start parsing a stream file by chunks...
        path_stream = self.path_stream
        with open(path_stream,'r') as fh:
            for line in fh:
                line = line.strip()

                m = regex_dict['geom'].search(line)
                if m is not None:
                    import pdb; pdb.set_trace()

                m = regex_dict['event_colon_dict'].match(line)
                if m is not None:
                    import pdb; pdb.set_trace()

                m = regex_dict['event_eq_dict'].match(line)
                if m is not None:
                    import pdb; pdb.set_trace()

                m = regex_dict['peak_found'].match(line)
                if m is not None:
                    import pdb; pdb.set_trace()

                m = regex_dict['peak_indexed'].match(line)
                if m is not None:
                    import pdb; pdb.set_trace()






path_stream = '/reg/data/ana15/cxi/cxig3514/scratch/cwang31/psocake/r0041/cxig3514_0041.stream'
## path_stream = '/reg/data/ana15/cxi/cxig3514/scratch/cwang31/psocake/r0041/cxig3514_0041_d100.stream'
## path_stream = '/reg/data/ana15/cxi/cxig3514/scratch/cwang31/psocake/r0041/test2.stream'
## path_stream = '/reg/data/ana15/cxi/cxig3514/scratch/cwang31/psocake/r0041/test4.stream'
## path_stream = '/reg/data/ana15/cxi/cxig3514/scratch/cwang31/psocake/r0041/test.stream'
## path_stream = '/reg/data/ana03/scratch/cwang31/pf/cxic0415_0101.test.stream'
config_stream_parser = ConfigParam( path_stream = path_stream )
stream_parser = StreamParser(config_stream_parser)
stream_parser.parse()
## peak_dict = stream_parser.peak_dict

## fl_peak_dict = 'peak_dict.pickle'
## with open(fl_peak_dict, 'wb') as fh:
##     pickle.dump(peak_dict, fh, protocol = pickle.HIGHEST_PROTOCOL)
