import torch

from mmcv.runner import BaseModule
from mmcv.cnn.bricks import ConvModule

from .concat_fuser import ConcatFuser
from ..utils import ConvModuleLN
from ..utils import EdgeResidual
from ..builder import FUSERS
from ..builder import build_attention

@FUSERS.register_module()
class AttentionFuser(BaseModule):
    """Fuse multi-modality features based on attention mechanism

    Args:
        attention_only (bool): Whether use attention only. If true, there will 
            be no conv module. Defaults to False.
        attention_cfg (dict | None): Config for attention module. If set to None,
            there will be no attention module.
    """
    def __init__(
            self,
            in_channels,
            out_channels=256,
            attention_only=False, # ONLY have attention module, no conv_module
            attention_cfg=dict(type="CBAM"),
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),):
        super(AttentionFuser, self).__init__()

        self.concat = ConcatFuser()
        self.attention_only = attention_only

        if not self.attention_only:
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.conv_module = EdgeResidual(in_channels, in_channels, in_channels,
                                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.conv1x1 = ConvModuleLN(in_channels, out_channels, 1, conv_cfg=conv_cfg, 
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)
            attention_cfg["in_channels"] = out_channels
        else:
            attention_cfg["in_channels"] = in_channels

        self.attention_cfg = attention_cfg
        if attention_cfg is not None:
            self.attn = build_attention(attention_cfg)

    def forward(self, features):
        """
            features (dict): Features output by encoders
                
                - "stream1": Torch.tensor, B,C,H,W
                - ...
        """
        # print('-------------------Attention Fuser------------------')
        # print([feature.shape for feature in features.values()])
        x = self.concat(features)

        if not self.attention_only:
            x = self.conv_module(x)
            x = self.conv1x1(x)

        if self.attention_cfg is not None:
            x = self.attn(x)
        
        return x
