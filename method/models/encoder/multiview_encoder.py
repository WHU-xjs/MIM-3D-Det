from typing import Any, Dict

import torch

from ..builder import (
    build_backbone,
    build_neck,
    build_vtransform,
)
from ..builder import ENCODERS
from .base_encoder import BaseEncoder

@ENCODERS.register_module()
class MultiViewEncoder(BaseEncoder):
    def __init__(
                self,
                backbone: Dict[str, Any],
                neck: Dict[str, Any] = None,
                vision_transform: Dict[str, Any] = None,
                stream_name="multiview"):
        super(MultiViewEncoder, self).__init__(stream_name)
        self.backbone = build_backbone(backbone)
        if neck is None:
            self.neck = None
        else:
            self.neck = build_neck(neck)
        self.vision_transform = build_vtransform(vision_transform)

    def forward(self, data, metas=None):
        """
        Args:
            data (dict): 
                "img": Torch.Tensor [B, N, C, H, W]
                "camera2lidar": 4x4 rigid transform matric, [B, N, 4, 4]
                "camera_intrinsics": 4x4 rigid transform matric, [B, N, 4, 4]
                "img_aug_matrix": 4x4 rigid transform matric, [B, N, 4, 4]
        """
        imgs = data["img"]
        # get [1, 1, N, C, H, W] when driv/lib ver mismatch, should no longer happen
        # print('multi-view encoder check', imgs.shape)
        # if imgs.dim() == 6 and imgs.size(0) == 1: imgs = imgs.squeeze(0)
        B, N, C, H, W = imgs.size()
        imgs = imgs.view(B * N, C, H, W)

        # resnet, fpn return tuples
        feats = self.backbone(imgs)
        if self.neck:
            feats_pyramid = self.neck(feats)

        if isinstance(feats_pyramid, torch.Tensor):
            feats_pyramid = [feats_pyramid] # Turn to list
        elif isinstance(feats_pyramid, tuple):
            feats_pyramid = list(feats_pyramid)
        # list is handled in transformer(now former encoder)
        # and better supports multi-level features

        for lvl, feat in enumerate(feats_pyramid):
            _, C, H, W = feat.size()
            feats_pyramid[lvl] = feat.view(B, N, C, H, W)
            # print('pyramid: ', feats_pyramid[lvl].size())

        # for LSSTransform
        geometries = {
            "camera2lidar": data["camera2lidar"],
            "camera_intrinsics": data["camera_intrinsics"],
            "img_aug_matrix": data["img_aug_matrix"],
        }
        feats_pyramid = self.vision_transform(feats_pyramid, geometries=geometries, img_metas=metas)

        return feats_pyramid