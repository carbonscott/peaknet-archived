#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

timestamp   = '2022_1028_1606_43'
drc_pickle  = 'label'
fl_pickle   = f'{timestamp}.pickle'
path_pickle = os.path.join(drc_pickle, fl_pickle)

with open(path_pickle, 'rb') as fh:
    data_pickle = pickle.load(fh)

data_list = data_pickle[0]
mask_dict = data_pickle[1]

rng_b, rng_e = 0, 226
rng_to_export = range(rng_b, rng_e)
for i in rng_to_export:
    img, label = data_list[i]
    mask       = mask_dict.get(i, None)

    if mask is not None: img *= mask

drc_dataset = 'datasets'
fl_dataset  = 'sfx.0002.npy'
path_dataset = os.path.join(drc_dataset, fl_dataset)
np.save(path_dataset, data_list[rng_b:rng_e])
