import torch
import torch.nn as nn
import numpy as np
from ..builder import ENCODERS, build_conv_group
from ..utils import EdgeResidual, SpatialTransformer
from .base_encoder import BaseEncoder
from collections import Sequence

import pickle

def gather_map(data):
    if isinstance(data["map_mask"], Sequence):
        map = torch.stack(data["map_mask"], dim=0)
    else:
        map = data["map_mask"]
    map = map.float().transpose(-1, -2)
    return map

@ENCODERS.register_module()
class PseudoMapEncoder(BaseEncoder):
    def __init__(self, stream_name="map"):
        super(PseudoMapEncoder, self).__init__(stream_name)

    def forward(self, data, metas=None):
        return gather_map(data)


@ENCODERS.register_module()
class ResizeMapEncoder(BaseEncoder):
    """Resize map size directly by interpolation.
    """
    def __init__(self, out_size, outofmap_layer=False, mode="bilinear", stream_name="map"):
        super(ResizeMapEncoder, self).__init__(stream_name)
        self.out_size = out_size
        self.outofmap_layer = outofmap_layer
        self.mode = mode

    def forward(self, data, metas=None):
        # hdmap (B, C, H, W)
        hdmap = gather_map(data)
        if self.outofmap_layer:
            layer = (~hdmap.any(axis=1)).unsqueeze(1)
            hdmap = torch.cat([hdmap, layer], axis=1)
        return torch.nn.functional.interpolate(hdmap, size=self.out_size, mode=self.mode)

@ENCODERS.register_module()
class ConvMapEncoder(BaseEncoder):
    """Convolution Backbone Map Encoder.

    Args:
        conv_group_cfg: (dict): The config of convolution group.

            - in_channels (int): The input channels of this module.
            - out_channels (list[int]): The output channels of this module.
            - mid_channels (list[int]): The input channels of the second convolution.
            - stride (list[int]): The stride of the first convolution. Defaults to 1.
            - kernel_size (int): The kernel size of the first convolution.
                Defaults to 3.
            - se_cfg (dict, optional): Config dict for se layer. Defaults to None,
                which means no se layer.
            - with_residual (bool): Use residual connection. Defaults to True.
            - conv_cfg (dict, optional): Config dict for convolution layer.
                Defaults to None, which means using conv2d.
            - norm_cfg (dict): Config dict for normalization layer.
                Defaults to ``dict(type='BN')``.
            - act_cfg (dict): Config dict for activation layer.
                Defaults to ``dict(type='ReLU')``.
            - drop_path_rate (float): stochastic depth rate. Defaults to 0.
            - with_cp (bool): Use checkpoint or not. Using checkpoint will save some
                memory while slowing down the training speed. Defaults to False.
            - init_cfg (dict | list[dict], optional): Initialization config dict.
        stream_name (str, optional): Stream name. Defaults to 'map'.
    """
    def __init__(
            self, 
            conv_group_cfg = dict(
                type="EdgeResidualGroup",
                input_channels=6,
                out_channels=[32, 64],
                mid_channels=[16, 32],
                strides=[2, 2],
                kernel_size=3,
                with_se=False,
                with_residual=True,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
                drop_path_rate=0.,
                with_cp=False,
                init_cfg=None
            ),
            outofmap_layer=False,
            stream_name="map"):
        super(ConvMapEncoder, self).__init__(stream_name)
        self.outofmap_layer = outofmap_layer
        if outofmap_layer: conv_group_cfg['input_channels'] += 1
        self.conv_group = build_conv_group(conv_group_cfg)

    def forward(self, data, metas=None):
        hdmap = gather_map(data)
        if self.outofmap_layer:
            # hdmap (B, C, H, W)
            layer = (~hdmap.any(axis=1)).unsqueeze(1)
            hdmap = torch.cat([hdmap, layer], axis=1)
        #@ too large to store all, used to calc stats
        pkl = './exp/nus/viz/curT.pkl'
        with open(pkl, 'wb') as hdmaps:
            pickle.dump(hdmap, hdmaps)

        x = self.conv_group(hdmap)
        return x

@ENCODERS.register_module()
class STMapEncoder(BaseEncoder):
    """Map Encoder with spatial transformer.

    Args:
        output_size: Output size of the interpolation when resizing feature 
            map. ONLY when feat_encoder is None.
        mode: Mode of the interpolation when resizing feature map. ONLY when 
            feat_encoder is None.
    """
    def __init__(
            self,
            spatial_transformer=dict(
                conv_group_cfg = dict(
                    type="EdgeResidualGroup",
                    input_channels=6,
                    out_channels=[32],
                    mid_channels=[16],
                    strides=[1]),
                mlp_hiddens = [16]
                ),
            feat_encoder=dict(
                type="EdgeResidualGroup",
                input_channels=6,
                out_channels=[32, 64],
                mid_channels=[16, 32],
                strides=[2, 2],
                kernel_size=3,
                with_se=False,
                with_residual=True,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
                drop_path_rate=0.,
                with_cp=False,
                init_cfg=None),
            output_size=(128, 128),
            mode="bilinear",
            stream_name="map"):
        super(STMapEncoder, self).__init__(stream_name)
        self.stnet = SpatialTransformer(**spatial_transformer)
        if feat_encoder is None:
            self.feat_encoder = None
        else:
            self.feat_encoder = build_conv_group(feat_encoder)
        self.output_size = output_size
        self.mode = mode

    def forward(self, data, metas=None):
        # print('----------------------Map Encoder-------------------')
        x = gather_map(data)
        x, _ = self.stnet(x)
        if self.feat_encoder:
            x = self.feat_encoder(x)
        else:
            x = torch.nn.functional.interpolate(
                x, size=self.output_size, mode=self.mode)
        # print('out map: ', x.shape)
        return x
            
