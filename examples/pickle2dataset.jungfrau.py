#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
## import pickle
from pickle5 import pickle
import numpy as np

basename =  "cxilz0720"

## timestamp   = "2022_1118_1306_22"    # Use the timestamp from the labeling applicaiton
timestamp   = "2023_0320_1514_03"    # Use the timestamp from the labeling applicaiton
drc_pickle  = "label"
fl_pickle   = f"jungfrau.{timestamp}.pickle"
path_pickle = os.path.join(drc_pickle, fl_pickle)

with open(path_pickle, 'rb') as fh:
    data_pickle = pickle.load(fh)

data_list = data_pickle[0]
mask_dict = data_pickle[1]

rng_b, rng_e = 0, 998
rng_to_export = range(rng_b, rng_e + 1)
for i in rng_to_export:
    img, label = data_list[i]
    mask       = mask_dict.get(i, None)

    if mask is not None: img *= mask

drc_dataset = f"datasets"
fl_dataset  = f"{basename}.0001.npy"
path_dataset = os.path.join(drc_dataset, fl_dataset)
np.save(path_dataset, data_list[rng_b:rng_e + 1])
