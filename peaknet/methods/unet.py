from peaknet.datasets.transform import center_crop
"""
---
title: U-Net
summary: >
    PyTorch implementation and tutorial of U-Net model.
---
# U-Net
This is an implementation of the U-Net model from the paper,
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://papers.labml.ai/paper/1505.04597).
U-Net consists of a contracting path and an expansive path.
The contracting path is a series of convolutional layers and pooling layers,
where the resolution of the feature map gets progressively reduced.
Expansive path is a series of up-sampling layers and convolutional layers
where the resolution of the feature map gets progressively increased.
At every step in the expansive path the corresponding feature map from the contracting path
concatenated with the current feature map.
![U-Net diagram from paper](unet.png)
Here is the [training code](experiment.html) for an experiment that trains a U-Net
on [Carvana dataset](carvana.html).
"""
import torch
## import torchvision.transforms.functional
from torch import nn


class DoubleConvolution(nn.Module):
    """
    ### Two $3 \times 3$ Convolution Layers
    Each step in the contraction path and expansive path have two $3 \times 3$
    convolutional layers followed by ReLU activations.
    In the U-Net paper they used $0$ padding,
    but we use $1$ padding so that final feature map is not cropped.
    """

    def __init__(self, in_channels, out_channels, uses_skip_connection = False):
        """
        :param in_channels: is the number of input channels
        :param out_channels: is the number of output channels
        """
        super().__init__()
        self.uses_skip_connection = uses_skip_connection

        # First $3 \times 3$ convolutional layer
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        # Second $3 \times 3$ convolutional layer
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()

        if uses_skip_connection:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(out_channels),
            )


    def forward(self, x: torch.Tensor):
        # Apply the two convolution layers and activations
        out = self.first(x)
        out = self.act1(out)
        out = self.second(out)
        out = self.act2(out)

        if self.uses_skip_connection:
            out += self.res(x)

        return out




class DownSample(nn.Module):
    """
    ### Down-sample
    Each step in the contracting path down-samples the feature map with
    a $2 \times 2$ max pooling layer.
    """

    def __init__(self):
        super().__init__()
        # Max pooling layer
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    """
    ### Up-sample
    Each step in the expansive path up-samples the feature map with
    a $2 \times 2$ up-convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Up-convolution
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class CropAndConcat(nn.Module):
    """
    ### Crop and Concatenate the feature map
    At every step in the expansive path the corresponding feature map from the contracting path
    concatenated with the current feature map.
    """
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        """
        :param x: current feature map in the expansive path
        :param contracting_x: corresponding feature map from the contracting path
        """

        # Crop the feature map from the contracting path to the size of the current feature map
        ## contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        contracting_x = center_crop(contracting_x, x.shape[2], x.shape[3])
        # Concatenate the feature maps
        x = torch.cat([x, contracting_x], dim=1)
        #
        return x


class UNet(nn.Module):
    """
    ## U-Net
    """
    ## def __init__(self, in_channels: int, out_channels: int, base_channels = 64, num_downsample_layer = None, requires_feature_map = False):
    def __init__(self, in_channels: int, out_channels: int, base_channels = 64, num_downsample_layer = None, uses_skip_connection = False):
        """
        :param in_channels: number of channels in the input image
        :param out_channels: number of channels in the result feature map
        """
        super().__init__()

        self.uses_skip_connection = uses_skip_connection


        # Double convolution layers for the contracting path.
        # The number of features gets doubled at each step starting from $64$.
        downsample_layer_list = [(in_channels, base_channels),
                                 (base_channels, base_channels * 2),
                                 (base_channels * 2, base_channels * 4),
                                 (base_channels * 4, base_channels * 8)]
        if num_downsample_layer is None: num_downsample_layer = len(downsample_layer_list)
        if isinstance(num_downsample_layer, int):
            num_downsample_layer = max(1, num_downsample_layer)
            num_downsample_layer = min(len(downsample_layer_list), num_downsample_layer)
        self.down_conv = nn.ModuleList([DoubleConvolution(i, o, uses_skip_connection = self.uses_skip_connection) for i, o in
                                        downsample_layer_list[:num_downsample_layer]
                                       ])
        # Down sampling layers for the contracting path
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])

        # The two convolution layers at the lowest resolution (the bottom of the U).
        self.middle_conv = DoubleConvolution(base_channels * 2**(num_downsample_layer-1), base_channels * 2**(num_downsample_layer), uses_skip_connection = self.uses_skip_connection)

        # Up sampling layers for the expansive path.
        # The number of features is halved with up-sampling.
        upsample_layer_list = [(base_channels * 16, base_channels * 8), 
                               (base_channels * 8,  base_channels * 4), 
                               (base_channels * 4,  base_channels * 2), 
                               (base_channels * 2,  base_channels)]
        size_upsample_layer_list = len(upsample_layer_list)
        num_upsample_layer = size_upsample_layer_list - num_downsample_layer
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
                                        upsample_layer_list[num_upsample_layer:]
                                       ])
        # Double convolution layers for the expansive path.
        # Their input is the concatenation of the current feature map and the feature map from the
        # contracting path. Therefore, the number of input features is double the number of features
        # from up-sampling.
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o, uses_skip_connection = self.uses_skip_connection) for i, o in
                                      upsample_layer_list[num_upsample_layer:]
                                     ])
        # Crop and concatenate layers for the expansive path.
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        # Final $1 \times 1$ convolution layer to produce the output
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        ## # Always keep the feature_map_dict for exporting purpose
        ## self.feature_map_dict = {}
        ## self.requires_feature_map = requires_feature_map

        ## # Switch to forward and track???
        ## if self.requires_feature_map:
        ##     self.forward = self.forward_and_track

    def forward(self, x: torch.Tensor):
        """
        :param x: input image
        """
        # To collect the outputs of contracting path for later concatenation with the expansive path.
        pass_through = []
        # Contracting path
        for i in range(len(self.down_conv)):
            # Two $3 \times 3$ convolutional layers
            x = self.down_conv[i](x)
            # Collect the output
            pass_through.append(x)
            # Down-sample
            x = self.down_sample[i](x)

        # Two $3 \times 3$ convolutional layers at the bottom of the U-Net
        x = self.middle_conv(x)

        # Expansive path
        for i in range(len(self.up_conv)):
            # Up-sample
            x = self.up_sample[i](x)
            # Concatenate the output of the contracting path
            x = self.concat[i](x, pass_through.pop())
            # Two $3 \times 3$ convolutional layers
            x = self.up_conv[i](x)

        # Final $1 \times 1$ convolution layer
        x = self.final_conv(x)

        return x
