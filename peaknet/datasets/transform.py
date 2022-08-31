#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def center_crop(img, size_y_crop, size_x_crop):
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
    y_min_img = (size_y_super - size_y_img) // 2
    x_min_img = (size_x_super - size_x_img) // 2
    img_super[...,
              y_min_img : y_min_img + size_y_img,
              x_min_img : x_min_img + size_x_img] = img

    # Crop...
    y_min_crop = (size_y_super - size_y_crop) // 2
    x_min_crop = (size_x_super - size_x_crop) // 2

    return img_super[...,
                     y_min_crop : y_min_crop + size_y_crop,
                     x_min_crop : x_min_crop + size_x_crop]




def coord_img_to_crop(coord_tuple, size_img_tuple, size_crop_tuple):
    # Unpack all inputs...
    y          , x           = coord_tuple
    size_y_img ,  size_x_img = size_img_tuple
    size_y_crop, size_x_crop = size_crop_tuple

    # Transform...
    y_crop = (size_y_crop - size_y_img) / 2 + y
    x_crop = (size_x_crop - size_x_img) / 2 + x

    y_crop += 0.5
    x_crop += 0.5

    y_crop = int(y_crop)
    x_crop = int(x_crop)

    ## # [NOT ENTIRELY SURE] Correct for one pixel...
    ## y_crop += 1
    ## x_crop += 1

    return y_crop, x_crop
