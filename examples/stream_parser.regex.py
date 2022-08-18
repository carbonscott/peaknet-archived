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
            (?>
                (?:
                    (?>
                        (?&FLOAT) # Match a floating number
                    )
                    \s+           # Match whitespace at least once
                ){4}              # Match the whole construct 4 times
            )

            (?&DET_PANEL)         # Match a detector panel

            (?(DEFINE)
                (?<FLOAT>
                    [-+]?         # Match a sign
                    (?>\d+)       # Match integer part
                    (?:\.\d*)?    # Match decimal part
                )
                (?<DET_PANEL>
                    (?: [0-9A-Za-z]+ $ )
                )
            )
            """
        )

        # Parse peaks found...
        regex_dict['peak_indexed'] = regex.compile( 
            r"""
            (?x)
            (?:
                (?>
                    (?&FLOAT) # Match a floating number
                )
                \s+           # Match whitespace at least once
            ){9}              # Match the whole construct 4 times

            (?&DET_PANEL)     # Match a detector panel

            (?(DEFINE)
                (?<FLOAT>
                    [-+]?         # Match a sign
                    (?>\d+)       # Match integer part
                    (?:\.\d*)?    # Match decimal part
                )
                (?<DET_PANEL>
                    (?: [0-9A-Za-z]+ $ )
                )
            )
            """
        )

        # Parse colon item in event...
        # DISCARD!!! Don't match absolute words with regex
        regex_dict['chunk_colon'] = regex.compile(
            r"""
            (?x)
            (?>
                (?P<KEY>     # Match a keyword below
                    (?:Image \s filename)
                |   (?:Event)
                )
            )
            :                # Match a colon
            (?P<VALUE>.+ $)  # Match the event number
            """
        )

        # Parse equal sign item in event...
        # DISCARD!!! Don't match absolute words with regex
        regex_dict['chunk_eq'] = regex.compile(
            r"""
            (?x)
            (?>
                (?P<KEY>     # Match a keyword below
                    indexed_by
                )
            )
            \s = \s          # Match a equal sign with blank spaces on both sides
            (?P<VALUE>.+ $)  # Match the event number
            """
        )

        # Parse detector gemoetry...
        regex_dict['geom'] = regex.compile(
            r"""
            (?x)
            # Match the pattern below
            (?> (?&DET_PANEL) ) / (?&COORD)
            \s = \s    # Match a equal sign with blank spaces on both sides
            (?&VALUE)  # Match the value of the coordinate

            (?(DEFINE)
                (?<DET_PANEL>
                    [0-9a-zA-Z]+
                )
                (?<COORD>
                    (?:min_fs)
                |   (?:min_ss)
                |   (?:max_fs)
                |   (?:max_ss)
                )
                (?<VALUE> [0-9]+ $)
            )
            """
        )

        return regex_dict


    def parse(self):
        '''
        Return a dictionary of matches that does ... .
        '''
        # Get all the regex for string matching...
        regex_dict = self.regex_dict

        # Keep results in match_dict...
        match_dict = {
            'geom'         : [],
            'chunk'        : {}
        }

        # Define some variables to hold values within a chunk...
        filename = None
        event_crystfel = None

        # State variable to decide if a chunk should be saved...
        save_chunk_ok = False

        # Parsing a stream file...
        path_stream = self.path_stream
        with open(path_stream,'r') as fh:
            for line in fh:
                # Strip out preceding or trailing blank spaces...
                line = line.strip()

                # Saving geom at the beginning of the file...
                # Match a geom object...
                m = regex_dict['geom'].match(line)
                if m is not None:
                    # Fetch values...
                    capture_dict = m.capturesdict()
                    panel = capture_dict['DET_PANEL'][0]
                    coord = capture_dict['COORD'][0]
                    value = capture_dict['VALUE'][0]

                    # Save values...
                    match_dict['geom'].append((panel, coord, int(value)))

                    continue

                # Match filename or image in a chunk
                m = regex_dict['chunk_colon'].match(line)
                if m is not None:
                    # Fetch values...
                    capture_dict = m.capturesdict()
                    k, v = capture_dict['KEY'][0], \
                           capture_dict['VALUE'][0]

                    # Save them temperarily...
                    if   k == 'Image filename':
                        filename = v
                        save_chunk_ok = False
                    elif k == 'Event':
                        event_crystfel = v[v.rfind('/') + 1:]
                        event_crystfel = int(event_crystfel)

                    continue

                # Match the indexed_by item...
                m = regex_dict['chunk_eq'].match(line)
                if m is not None:
                    # Fetch values...
                    capture_dict = m.capturesdict()
                    k, v = capture_dict['KEY'][0], \
                           capture_dict['VALUE'][0]

                    # Is this chunk indexed???
                    if v != 'none':
                        save_chunk_ok = True
                        continue

                # This chunk is indexed...
                if save_chunk_ok:
                    # Initialize data structure (dict) for saving information...
                    if not filename in match_dict['chunk']:
                        match_dict['chunk'][filename] = {}

                    if not event_crystfel in match_dict['chunk'][filename]:
                        match_dict['chunk'][filename][event_crystfel] = {
                            'peak_found'   : {},
                            'peak_indexed' : {},
                        }

                    # Match found peaks...
                    m = regex_dict['peak_found'].match(line)
                    if m is not None:
                        # Fetch values...
                        capture_dict = m.capturesdict()
                        x = float(capture_dict['FLOAT'][0])
                        y = float(capture_dict['FLOAT'][1])
                        panel = capture_dict['DET_PANEL'][0]

                        # Save values...
                        if  not panel in               \
                            match_dict['chunk']        \
                                      [filename]       \
                                      [event_crystfel] \
                                      ['peak_found']:
                            match_dict['chunk']        \
                                      [filename]       \
                                      [event_crystfel] \
                                      ['peak_found']   \
                                      [panel] = [(x, y)]
                        else:
                            match_dict['chunk']        \
                                      [filename]       \
                                      [event_crystfel] \
                                      ['peak_found']   \
                                      [panel].append((x, y))

                        continue

                    # Match indexed peaks...
                    m = regex_dict['peak_indexed'].match(line)
                    if m is not None:
                        # Fetch values...
                        capture_dict = m.capturesdict()
                        x = float(capture_dict['FLOAT'][-2])
                        y = float(capture_dict['FLOAT'][-1])
                        panel = capture_dict['DET_PANEL'][0]

                        # Save values...
                        if  not panel in               \
                            match_dict['chunk']        \
                                      [filename]       \
                                      [event_crystfel] \
                                      ['peak_indexed']:
                            match_dict['chunk']        \
                                      [filename]       \
                                      [event_crystfel] \
                                      ['peak_indexed']   \
                                      [panel] = [(x, y)]
                        else:
                            match_dict['chunk']        \
                                      [filename]       \
                                      [event_crystfel] \
                                      ['peak_indexed']   \
                                      [panel].append((x, y))

        self.match_dict = match_dict




path_stream = '/reg/data/ana15/cxi/cxig3514/scratch/cwang31/psocake/r0041/cxig3514_0041.stream'
## path_stream = '/reg/data/ana15/cxi/cxig3514/scratch/cwang31/psocake/r0041/cxig3514_0041_d100.stream'
## path_stream = '/reg/data/ana15/cxi/cxig3514/scratch/cwang31/psocake/r0041/test2.stream'
## path_stream = '/reg/data/ana15/cxi/cxig3514/scratch/cwang31/psocake/r0041/test4.stream'
## path_stream = '/reg/data/ana15/cxi/cxig3514/scratch/cwang31/psocake/r0041/test.stream'
## path_stream = '/reg/data/ana03/scratch/cwang31/pf/cxic0415_0101.test.stream'
## path_stream = '/reg/data/ana03/scratch/cwang31/pf/streams/test.stream'
## path_stream = '/reg/data/ana03/scratch/cwang31/pf/streams/multicrystal.stream'
config_stream_parser = ConfigParam( path_stream = path_stream )
stream_parser = StreamParser(config_stream_parser)
stream_parser.parse()
## peak_dict = stream_parser.peak_dict

## fl_peak_dict = 'peak_dict.pickle'
## with open(fl_peak_dict, 'wb') as fh:
##     pickle.dump(peak_dict, fh, protocol = pickle.HIGHEST_PROTOCOL)
