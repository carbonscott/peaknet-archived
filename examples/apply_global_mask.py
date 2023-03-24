#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pickle5 import pickle


timestamp        = "2023_0320_1514_03"
fl_mask_pickle   = f"masked.jungfrau.{timestamp}.pickle"
path_mask_pickle = f"label/{fl_mask_pickle}"
with open(path_mask_pickle, 'rb') as handle:
    data = pickle.load(handle)

idx = 2
global_mask = data[1][idx]

for k in range(len(data[0])):
    print(k)
    if k == idx: continue

    if k not in data[1]: data[1][k] = None
    data[1][k] = global_mask.copy()

fl_out_pickle   = f"jungfrau.{timestamp}.pickle"
path_out_pickle = f"label/{fl_out_pickle}"
with open(path_out_pickle, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
