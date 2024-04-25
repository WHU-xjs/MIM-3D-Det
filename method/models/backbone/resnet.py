from typing import List, Tuple

import torch
from torch import nn

import torch.utils.checkpoint as checkpoint
from mmdet.models.backbones.resnet import Bottleneck, BasicBlock
from mmcv.runner import BaseModule
from ..builder import BACKBONES

# mmdet build_norm_layer always uses out_channel as norm channel
# conficts because conv module require (B,C,H,W) but LN require (*,C)
class BasicBlockLN(BasicBlock):
    """default module using LN, permute before and after norm"""
    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = out.permute(0,2,3,1)
            out = self.norm1(out)
            out = out.permute(0,3,1,2)
            out = self.relu(out)

            out = self.conv2(out)
            out = out.permute(0,2,3,1)
            out = self.norm2(out)
            out = out.permute(0,3,1,2)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = checkpoint.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

class BottleneckLN(Bottleneck):
    """default module using LN, permute before and after norm"""
    def forward(self, x):

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = out.permute(0,2,3,1)
            out = self.norm1(out)
            out = out.permute(0,3,1,2)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = out.permute(0,2,3,1)
            out = self.norm2(out)
            out = out.permute(0,3,1,2)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = out.permute(0,2,3,1)
            out = self.norm3(out)
            out = out.permute(0,3,1,2)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = checkpoint.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

@BACKBONES.register_module()
class ResNetForBEVDet(BaseModule):
    def __init__(self, 
                 numC_input, 
                 numC_middle=None,
                 num_layer=[2,2,2], 
                 num_channels=None, 
                 stride=[2,2,2],
                 backbone_output_ids=None, 
                 norm_cfg=dict(type='BN'),
                 with_cp=False, 
                 block_type='Basic',):
        super(ResNetForBEVDet, self).__init__()
        # build backbone
        # assert len(num_layer)>=3
        assert len(num_layer)==len(stride)
        if numC_middle is None:
            numC_middle = numC_input
        num_channels = [numC_middle*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if norm_cfg is not None and norm_cfg['type'] == 'LN':
            basicblock = BasicBlockLN
            bottleneck = BottleneckLN
        else:
            basicblock = BasicBlock
            bottleneck = Bottleneck
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer=[bottleneck(curr_numC, num_channels[i]//4, stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC,num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                curr_numC= num_channels[i]
                layer.extend([bottleneck(curr_numC, curr_numC//4,
                                         norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer=[basicblock(curr_numC, num_channels[i], stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC,num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                curr_numC= num_channels[i]
                layer.extend([basicblock(curr_numC, curr_numC, norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats
