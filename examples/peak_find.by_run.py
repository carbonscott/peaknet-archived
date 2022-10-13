#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
import h5py

import time

from scipy import ndimage

from peaknet.datasets.utils           import PsanaImg
## from peaknet.methods.unet             import UNet
from peaknet.methods.unet_simple      import UNet
from peaknet.model                    import ConfigPeakFinderModel, PeakFinderModel
from peaknet.datasets.stream_parser   import GeomInterpreter
from peaknet.datasets.transform       import center_crop, coord_img_to_crop, coord_crop_to_img

class ConfigPeakFinder:
    ''' Biolerplate code to config dataset classs'''

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items(): setattr(self, k, v)


    def report(self):
        logger.info(f"___/ Configure Dataset \___")
        for k, v in self.__dict__.items():
            if k == 'kwargs': continue
            logger.info(f"KV - {k:16s} : {v}")


class PsanaPeakFinder:
    def __init__(self, config):
        self.model         = getattr(config, 'model'        , None)
        self.path_chkpt    = getattr(config, 'path_chkpt'   , None)
        self.exp           = getattr(config, 'exp'          , None)
        self.run           = getattr(config, 'run'          , None)
        self.access_mode   = getattr(config, 'access_mode'  , None)
        self.detector_name = getattr(config, 'detector_name', None)
        self.drc           = getattr(config, 'drc'          , None)
        self.event_list    = getattr(config, 'event_list'   , None)
        self.fl_cxi        = getattr(config, 'fl_cxi'       , None)
        self.photon_energy = getattr(config, 'photon_energy', None)
        self.encoder_value = getattr(config, 'encoder_value', None)
        self.adu_threshold = getattr(config, 'adu_threshold', None)

        self.psana_img = PsanaImg(exp, run, access_mode, detector_name)

        self.multipanel_mask = self.psana_img.create_bad_pixel_mask()

        # Load model to gpus if available...
        self.device = 'cpu'
        if self.path_chkpt is not None and torch.cuda.is_available():
            self.device = torch.cuda.current_device()

            chkpt = torch.load(self.path_chkpt)
            self.model.load_state_dict(chkpt)
            self.model = torch.nn.DataParallel(self.model.method).to(self.device)

        # Turn on inference model...
        self.model.eval()

        return None


    def fetch_img_per_event(self, event):
        # Load image...
        multipanel_img = self.psana_img.get(event, None, 'calib')
        multipanel_img_masked = self.multipanel_mask * multipanel_img
        img = self.convert_psana_to_cheetah(multipanel_img_masked)

        return img


    def find_peak_per_event(self, event):
        # Load image...
        img = self.fetch_img_per_event(event)
        img = img[None, ]
        ## img = img[:, None, ]

        # Add a fake batch dim and move the data to gpu if available...
        # The model is trained with the extra dimension
        batch_img  = torch.Tensor(img[None,])
        ## batch_img  = torch.Tensor(img)
        batch_img  = batch_img.to(self.device)

        # Find the predicted mask...
        with torch.no_grad():
            batch_mask_predicted = self.model.forward(batch_img)

        # Fetch the offset for coordinate conversion downstream...
        size_y_crop, size_x_crop = batch_mask_predicted.shape[-2:]
        _, offset_tuple = center_crop(batch_img, size_y_crop, size_x_crop, return_offset_ok = True)

        # Save them to cpu...
        batch_mask_predicted = batch_mask_predicted.cpu().detach().numpy()
        ## mask_predicted       = batch_mask_predicted.reshape(*batch_mask_predicted.shape[-3:])    # Remove fake batch layer
        mask_predicted       = np.squeeze(batch_mask_predicted, axis = 1)    # Remove fake batch layer

        # Remove the extra dimension required for inference
        mask_predicted = mask_predicted[0]

        # Thresholding...
        adu_threshold = self.adu_threshold
        mask = mask_predicted.copy()
        mask[  mask_predicted < adu_threshold ] = 0
        ## mask[~(mask_predicted < adu_threshold)] = 1

        # Put box on peaks...
        ## structure = np.ones((3, 3), dtype=bool)
        ## peak_labeled, num_peak = ndimage.label(mask, structure)
        peak_labeled, num_peak = ndimage.label(mask)
        peak_pos_list = ndimage.center_of_mass(mask, peak_labeled, range(num_peak))

        # Convert coordinates...
        peak_pos_list_converted = []
        size_y_img, size_x_img = img.shape[-2:]
        for y, x in peak_pos_list:
            if np.isnan(y): continue

            y_img, x_img = coord_crop_to_img((y, x), 
                                             (size_y_img, size_x_img), 
                                             (size_y_crop, size_x_crop),
                                             offset_tuple)
            peak_pos_list_converted.append((y_img, x_img))

        return peak_pos_list_converted


    def fetch_batch_img(self, event_list):
        # Initialize matrix to hold all images from the event list...
        num_event = len(event_list)
        size_y, size_x = self.fetch_img_per_event(event_list[0]).shape
        batch_img = np.zeros((num_event, size_y, size_x))

        for i, event in enumerate(event_list):
            batch_img[i] = self.fetch_img_per_event(event)

        return batch_img


    def find_batch_peak(self, event_list, min_num_peak_required = 15):
        ''' Max event to process at once should be addressed in event splitting
            stage.  This function assumes all events in the input list will be
            subject to matrix computaiton in GPU at once.
        '''
        # Load image...
        batch_img = self.fetch_batch_img(event_list)

        # Move the data to gpu if available...
        # Add a fake channel dim at the dim=1 position
        batch_img  = torch.Tensor(batch_img[:, None])
        batch_img  = batch_img.to(self.device)

        # Find the predicted mask...
        with torch.no_grad():
            batch_mask_predicted = self.model.forward(batch_img)

        # Fetch the offset for coordinate conversion downstream...
        size_y_crop, size_x_crop = batch_mask_predicted.shape[-2:]
        _, offset_tuple = center_crop(batch_img, size_y_crop, size_x_crop, return_offset_ok = True)

        # Save them to cpu...
        batch_mask_predicted = batch_mask_predicted.cpu().detach().numpy()

        # Sequeeze the channel dimension (dim = 1)...
        batch_mask_predicted = batch_mask_predicted[:, 0]

        # Thresholding...
        adu_threshold = self.adu_threshold
        batch_mask = batch_mask_predicted.copy()
        batch_mask[  batch_mask_predicted < adu_threshold ] = 0

        # Put box on peaks...
        batch_peak_list = self.calc_batch_center_of_mass(batch_mask)
        peak_list, seqi_filter_list = self.filter_batch_peak(batch_peak_list, min_num_peak_required = min_num_peak_required)

        # Convert coordinates...
        peak_list_converted = []
        size_y_img, size_x_img = batch_img.shape[-2:]
        for peak_per_event_list in peak_list:
            peak_per_event_list_converted = []
            for y, x in peak_per_event_list:
                if np.isnan(y): continue

                y_img, x_img = coord_crop_to_img((y, x), 
                                                 (size_y_img, size_x_img), 
                                                 (size_y_crop, size_x_crop),
                                                 offset_tuple)
                peak_per_event_list_converted.append((y_img, x_img))
            peak_list_converted.append(peak_per_event_list_converted)

        return peak_list_converted, seqi_filter_list


    def combine_batch_peak_to_dict(self, batch_peak_list):
        peak_dict = {}
        for idx_in_batch, y, x in batch_peak_list:
            if np.isnan(y): continue

            idx_in_batch = int(idx_in_batch)

            if not idx_in_batch in peak_dict: peak_dict[idx_in_batch] = []

            peak_dict[idx_in_batch].append((y, x))

        return peak_dict


    def filter_batch_peak(self, batch_peak_list, min_num_peak_required = 15):
        peak_dict = self.combine_batch_peak_to_dict(batch_peak_list)

        peak_list = [ v for k, v in peak_dict.items() if len(v) > min_num_peak_required ]
        seqi_filter_list = [ int(k) for k, v in peak_dict.items() if len(v) > min_num_peak_required ]

        return peak_list, seqi_filter_list


    def calc_batch_center_of_mass(self, batch_mask):
        # Set up structure to find connected component in 2D only...
        structure = np.zeros((3, 3, 3))
        #                     ^  ^^^^
        # batch_______________|   |
        #                         |
        # 2D image________________|

        # Define structure in 2D image at the middle layer
        structure[1] = np.array([[0,1,0],
                                 [1,1,1],
                                 [0,1,0]])

        # Fetch labels...
        batch_label, batch_num_feature = ndimage.label(batch_mask, structure = structure)

        # Calculate batch center of mass...
        ## batch_center_of_mass = ndimage.center_of_mass(batch_mask, batch_label, range(batch_num_feature))
        batch_center_of_mass = ndimage.maximum_position(batch_mask, batch_label, range(batch_num_feature))

        return batch_center_of_mass


    def convert_psana_to_cheetah(self, panel_list):
        # [!!!] Hard code
        dim0 = 8 * 185
        dim1 = 4 * 388

        # Convert calib image to cheetah image
        img = np.zeros((dim0, dim1))
        counter = 0
        for quad in range(4):
            for seg in range(8):
                img[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = panel_list[counter, :, :]
                counter += 1

        return img


    def save_batch_peak_to_cxi(self, event_list, max_num_peak = 2048):
        peak_list, seqi_filter_list = self.find_batch_peak(event_list)
        num_event = len(peak_list)

        photon_energy = self.photon_energy
        fl_cxi        = self.fl_cxi

        with h5py.File(fl_cxi, 'w') as myHdf5:
            # [!!!] Hard code
            dim0 = 8 * 185
            dim1 = 4 * 388

            grpName     = "/entry_1/result_1"
            dset_nPeaks = "/nPeaks"
            dset_posX   = "/peakXPosRaw"
            dset_posY   = "/peakYPosRaw"
            dset_atot   = "/peakTotalIntensity"

            grp = myHdf5.create_group(grpName)
            myHdf5.create_dataset(grpName + dset_nPeaks, (num_event,             ), dtype='int')
            myHdf5.create_dataset(grpName + dset_posX  , (num_event, max_num_peak), dtype='float32', chunks=(1, max_num_peak))
            myHdf5.create_dataset(grpName + dset_posY  , (num_event, max_num_peak), dtype='float32', chunks=(1, max_num_peak))
            myHdf5.create_dataset(grpName + dset_atot  , (num_event, max_num_peak), dtype='float32', chunks=(1, max_num_peak))

            myHdf5.create_dataset("/LCLS/detector_1/EncoderValue", (1,), dtype=float)
            myHdf5.create_dataset("/LCLS/photon_energy_eV", (1,), dtype=float)
            dset = myHdf5.create_dataset("/entry_1/data_1/data", (num_event, dim0, dim1), dtype=np.float32) # change to float32
            ## dsetM = myHdf5.create_dataset("/entry_1/data_1/mask", (dim0, dim1), dtype='int')

            # Save images...
            # [IMPROVE]
            event_filter_list = [ event_list[i] for i in seqi_filter_list ]
            dset[:] = self.fetch_batch_img(event_filter_list)

            # [QUESTION] Pixel shift by 0.5
            for seqi, peak_per_event_list in enumerate(peak_list):
                nPeaks = len(peak_per_event_list)
                for i, peak in enumerate(peak_per_event_list):
                    cheetahRow, cheetahCol = peak
                    myHdf5[grpName + dset_posX][seqi, i] = cheetahCol
                    myHdf5[grpName + dset_posY][seqi, i] = cheetahRow
                myHdf5[grpName + dset_nPeaks][seqi] = nPeaks

            myHdf5["/LCLS/detector_1/EncoderValue"][0] = self.encoder_value  # mm
            myHdf5["/LCLS/photon_energy_eV"][0] = photon_energy


    def save_peak_to_cxi(self, event_list, min_num_peak = 30, max_num_peak = 2048):
        # Process all if no event is specified...
        if not len(event_list):
            event_list = range(len(self.psana_img.timestamps))

        peak_list = []
        event_filtered_list = []
        for event in event_list:
            print(f"Processing {event}...")

            time_start = time.time()
            peak_per_event_list = self.find_peak_per_event(event)
            time_end = time.time()

            time_delta = time_end - time_start
            print(f"Time delta: {time_delta} second.")

            if len(peak_per_event_list) < min_num_peak: continue

            peak_list.append(peak_per_event_list)
            event_filtered_list.append(event)
        num_event = len(peak_list)

        photon_energy = self.photon_energy
        fl_cxi        = self.fl_cxi

        with h5py.File(fl_cxi, 'w') as myHdf5:
            # [!!!] Hard code
            dim0 = 8 * 185
            dim1 = 4 * 388

            grpName     = "/entry_1/result_1"
            dset_nPeaks = "/nPeaks"
            dset_posX   = "/peakXPosRaw"
            dset_posY   = "/peakYPosRaw"
            dset_atot   = "/peakTotalIntensity"

            grp = myHdf5.create_group(grpName)
            myHdf5.create_dataset(grpName + dset_nPeaks, (num_event,             ), dtype='int')
            myHdf5.create_dataset(grpName + dset_posX  , (num_event, max_num_peak), dtype='float32', chunks=(1, max_num_peak))
            myHdf5.create_dataset(grpName + dset_posY  , (num_event, max_num_peak), dtype='float32', chunks=(1, max_num_peak))
            myHdf5.create_dataset(grpName + dset_atot  , (num_event, max_num_peak), dtype='float32', chunks=(1, max_num_peak))

            myHdf5.create_dataset("/LCLS/detector_1/EncoderValue", (1,), dtype=float)
            myHdf5.create_dataset("/LCLS/photon_energy_eV", (1,), dtype=float)
            dset = myHdf5.create_dataset("/entry_1/data_1/data", (num_event, dim0, dim1), dtype=np.float32) # change to float32
            ## dsetM = myHdf5.create_dataset("/entry_1/data_1/mask", (dim0, dim1), dtype='int')

            # Save images...
            # [IMPROVE]
            ## dset[:] = self.fetch_batch_img(event_filtered_list)
            for seqi, event in enumerate(event_filtered_list):
                dset[seqi] = self.fetch_img_per_event(event)

            # [QUESTION] Pixel shift by 0.5
            for seqi, peak_per_event_list in enumerate(peak_list):
                nPeaks = len(peak_per_event_list)
                for i, peak in enumerate(peak_per_event_list):
                    cheetahRow, cheetahCol = peak
                    myHdf5[grpName + dset_posX][seqi, i] = cheetahCol
                    myHdf5[grpName + dset_posY][seqi, i] = cheetahRow
                myHdf5[grpName + dset_nPeaks][seqi] = nPeaks

            myHdf5["/LCLS/detector_1/EncoderValue"][0] = self.encoder_value  # mm
            myHdf5["/LCLS/photon_energy_eV"][0] = photon_energy

        # Provide lst file for indexing...
        # [IMPROVE] Move this feature to elsewhere
        # Work on one cxi at a time
        path_cxi = os.path.join(self.drc, fl_cxi)
        basename = fl_cxi[:fl_cxi.rfind('.')]
        fl_lst = f'{basename}.lst'
        with open(fl_lst,'w') as fh:
            for i, event in enumerate(event_filtered_list):
                fh.write(f"{path_cxi} //{i}")
                fh.write("\n")


    def save_peak_per_event(self, event):
        peak_list = self.find_peak_per_event(event)

        photon_energy = self.photon_energy
        fl_cxi        = self.fl_cxi

        with h5py.File(fl_cxi, 'w') as myHdf5:
            # [!!!] Hard code
            dim0 = 8 * 185
            dim1 = 4 * 388

            grpName     = "/entry_1/result_1"
            dset_nPeaks = "/nPeaks"
            dset_posX   = "/peakXPosRaw"
            dset_posY   = "/peakYPosRaw"
            dset_atot   = "/peakTotalIntensity"

            # [!!!] Hard code
            max_num_peak = 2048

            grp = myHdf5.create_group(grpName)
            myHdf5.create_dataset(grpName + dset_nPeaks, (1,            ), dtype='int')
            myHdf5.create_dataset(grpName + dset_posX  , (1, max_num_peak), dtype='float32', chunks=(1, max_num_peak))
            myHdf5.create_dataset(grpName + dset_posY  , (1, max_num_peak), dtype='float32', chunks=(1, max_num_peak))
            myHdf5.create_dataset(grpName + dset_atot  , (1, max_num_peak), dtype='float32', chunks=(1, max_num_peak))

            myHdf5.create_dataset("/LCLS/detector_1/EncoderValue", (1,), dtype=float)
            myHdf5.create_dataset("/LCLS/photon_energy_eV", (1,), dtype=float)
            dset = myHdf5.create_dataset("/entry_1/data_1/data", (1, dim0, dim1), dtype=np.float32) # change to float32
            ## dsetM = myHdf5.create_dataset("/entry_1/data_1/mask", (dim0, dim1), dtype='int')

            img = self.img
            dset[0, :, :] = img

            nPeaks = len(peak_list)
            for i, peak in enumerate(peak_list):
                cheetahRow, cheetahCol = peak
                myHdf5[grpName + dset_posX][0, i] = cheetahCol
                myHdf5[grpName + dset_posY][0, i] = cheetahRow
            myHdf5[grpName + dset_nPeaks][0] = nPeaks

            myHdf5["/LCLS/detector_1/EncoderValue"][0] = self.encoder_value  # mm
            myHdf5["/LCLS/photon_energy_eV"][0] = photon_energy


def convert_peaks_to_cheetah(s, r, c) :
    """Converts psana seg, row, col assuming (32,185,388)
       to cheetah 2-d table row and col (8*185, 4*388)
    """
    segs, rows, cols = (32,185,388)
    row2d = (int(s)%8) * rows + int(r) # where s%8 is a segment in quad number [0,7]
    col2d = (int(s)/8) * cols + int(c) # where s/8 is a quad number [0,3]
    return row2d, col2d

def convert_peaks_to_psana(row2d, col2d) :
    """Converts cheetah 2-d table row and col (8*185, 4*388)
       to psana seg, row, col assuming (32,185,388)
    """
    if isinstance(row2d, np.ndarray):
        row2d = row2d.astype('int')
        col2d = col2d.astype('int')
    segs, rows, cols = (32,185,388)
    s = (row2d / rows) + (col2d / cols * 8)
    r = row2d % rows
    c = col2d % cols
    return s, r, c


## timestamp = "2022_0908_1013_52"    # frac_train = 0.5, pos_weight = 2, pretty good.
timestamp = "2022_1012_1131_43"    # frac_train = 0.5, pos_weight = 2, pretty good.

base_channels = 8

exp           = 'cxic0415'
run           = 101

event_list    = []

photon_energy = 12688.890590380644    # eV
encoder_value = -450.0034
adu_threshold = 100

access_mode   = 'idx'
detector_name = 'CxiDs1.0:Cspad.0'
## fl_cxi        = f'pf.{exp}.{run}.{timestamp}.center_of_mass.cxi'
fl_cxi        = f'pf.{exp}.{run}.{timestamp}.maxpos.cxi'

# Config the model...
method = UNet(in_channels = 1, out_channels = 1, base_channels = base_channels)
pos_weight = 2.0
config_peakfinder = ConfigPeakFinderModel( method     = method,
                                           pos_weight = pos_weight, )
model = PeakFinderModel(config_peakfinder)

# Initialize weights...
def init_weights(module):
    if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
        module.weight.data.normal_(mean = 0.0, std = 0.02)
model.apply(init_weights)

# Define chkpt...
drc_cwd          = os.getcwd()
DRCCHKPT         = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
if not os.path.exists(prefixpath_chkpt): os.makedirs(prefixpath_chkpt)
path_chkpt = None
if timestamp is not None:
    fl_chkpt   = f"{timestamp}.train.chkpt"
    path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)

# Config peak finder...
config_pf = ConfigPeakFinder( model         = model,
                              path_chkpt    = path_chkpt, 
                              exp           = exp,
                              run           = run,
                              access_mode   = access_mode,
                              detector_name = detector_name, 
                              drc           = drc_cwd,
                              fl_cxi        = fl_cxi,
                              photon_energy = photon_energy,
                              encoder_value = encoder_value,
                              adu_threshold = adu_threshold, )
pf = PsanaPeakFinder(config_pf)
pf.save_peak_to_cxi(event_list = event_list, min_num_peak = 15)
