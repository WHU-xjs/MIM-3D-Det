from ..builder import FUSERS
from mmcv.runner import BaseModule
import torch

@FUSERS.register_module()
class AddFuser(BaseModule):
    def __init__(self):
        super(AddFuser, self).__init__()

    def forward(self, features):
        """
            features (dict): Features output by encoders
                
                - "stream1": Torch.tensor, B,C,H,W
                - ...
        """
        feat_maps = list(features.values())
        
        # Check shape. B, C, H, W of each feature map should be same.
        B0, C0, H0, W0 = feat_maps[0].shape
        feat = False
        for fm in feat_maps:
            B, C, H, W = fm.shape
            assert (B0 == B and C0 == C and H0 == H and W0 == W), \
                "Shape of feature maps must be same."
            if isinstance(feat, bool): feat = fm
            else: feat += fm

        return feat