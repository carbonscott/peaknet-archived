#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def center_crop(img, size_y_crop, size_x_crop, return_offset_ok = False):
    '''
    Return the cropped area and associated offset for coordinate transformation
    purposes.  
    '''
    # Get metadata (device)...
    device = img.device

    # Get the size the original image...
    # It might have other dimensions
    dim = img.shape
    size_y_img, size_x_img = dim[-2:]
    dim_storage = dim[:-2]

    # Initialize the super image that covers both the image and the crop...
    size_y_super = max(size_y_img, size_y_crop)
    size_x_super = max(size_x_img, size_x_crop)
    img_super = torch.zeros((*dim_storage, size_y_super, size_x_super), device = device)

    # Paint the original image onto the super image...
    # Area
    y_min_img = (size_y_super - size_y_img) // 2
    x_min_img = (size_x_super - size_x_img) // 2
    img_super[...,
              y_min_img : y_min_img + size_y_img,
              x_min_img : x_min_img + size_x_img] = img

    # Find crop range...
    # Min
    y_min_crop = (size_y_super - size_y_crop) // 2
    x_min_crop = (size_x_super - size_x_crop) // 2

    # Max
    y_max_crop = y_min_crop + size_y_crop
    x_max_crop = x_min_crop + size_x_crop

    # Crop the image area...
    img_crop =  img_super[...,
                          y_min_crop : y_max_crop,
                          x_min_crop : x_max_crop]

    # Pack things to return in a tuple
    ret_tuple = img_crop
    if return_offset_ok:
        # Min float for finding the offset
        y_min_crop_float = (size_y_super - size_y_crop) / 2
        x_min_crop_float = (size_x_super - size_x_crop) / 2

        # Offset introduced due to the integer division
        offset_tuple = (y_min_crop_float - y_min_crop, x_min_crop_float - x_min_crop)

        ret_tuple = (img_crop, offset_tuple)

    return ret_tuple




def coord_img_to_crop(coord_tuple, size_img_tuple, size_crop_tuple, offset_tuple = ()):
    '''
    Need some unit test.
    '''
    # Unpack all inputs...
    y          , x           = coord_tuple
    size_y_img ,  size_x_img = size_img_tuple
    size_y_crop, size_x_crop = size_crop_tuple

    # Transform...
    y_crop = (size_y_crop - size_y_img) / 2 + y
    x_crop = (size_x_crop - size_x_img) / 2 + x

    if len(offset_tuple) == 2:
        y_offset, x_offset = offset_tuple
        y_crop += y_offset
        x_crop += x_offset

    return y_crop, x_crop
