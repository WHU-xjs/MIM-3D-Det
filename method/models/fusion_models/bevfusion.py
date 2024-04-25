from typing import Any, Dict, List, Optional
from mmdet3d.core import bbox3d2result
from .base import Base3DFusionModel
from ..builder import MODELS
from ..builder import build_encoder, build_fuser, build_decoder, build_head
from ..utils import visual
from collections import Counter
from mmcv.runner import ModuleList

@MODELS.register_module()
class BEVFusion(Base3DFusionModel):
    """BEVFusion.

    Args:
        encoders (list): List of encoders' config.
        fuser (dict): Config of fuser.
        decoder (dict): Config of decoder.
        head (dict): Config of head.
        train_cfg (dict)
        test_cfg (dict)
    """

    def __init__(
        self,
        encoders: List[Dict],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        head: Dict[str, Any],
        train_cfg = None,
        test_cfg = None
        ):
        super(BEVFusion, self).__init__()

        self.encoders = []
        for encoder in encoders:
            self.encoders.append(build_encoder(encoder))
        self.encoders = ModuleList(self.encoders)
        
        stream_names = Counter([ed.stream_name for ed in self.encoders])
        assert stream_names.most_common(1)[0][1] == 1, \
            "The names of encoders should be unique."

        self.fuser = build_fuser(fuser)
        self.decoder = build_decoder(decoder)

        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)

        # self.sample_counter = 0

    def extract_feature(self, input_data, metas):
        """
            input_data (dict): sensors' data
                
                - "imgs"
                - "map_mask"
        """
        features = {}
        for encoder in self.encoders:
            features[encoder.stream_name] = encoder(input_data, metas)
        
        # @ my visualization
        # vispath = './exp/nus/visualize/bevdet-tiny-conv-bam/msk/'
        # visual.save_vis_bev(features, vispath, self.sample_counter, (4,4))
        # self.sample_counter += 1

        fuse = self.fuser(features)

        x = self.decoder(fuse)
        return x

    def forward_train(self, input_data, metas, gt_labels, gt_bboxes_ignore=None):
        losses = dict()

        # extract feature before head
        feat = self.extract_feature(input_data, metas)
        outs = self.head(feat)

        # The loss() interfaces of the heads in the mmdetection3d are not unified
        if self.head.__class__.__name__ == "Anchor3DHead":
            loss_inputs = [*outs, gt_labels['gt_bboxes_3d'], gt_labels['gt_labels_3d'], metas]
        else:
            loss_inputs = [gt_labels['gt_bboxes_3d'], gt_labels['gt_labels_3d'], outs]
        loss_bbox_head = self.head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
        losses.update(loss_bbox_head)
        return losses

    def simple_test(self, input_data, metas, rescale=False):
        feat = self.extract_feature(input_data, metas)
        outs = self.head(feat)

        if self.head.__class__.__name__ == "Anchor3DHead":
            bbox_list = self.head.get_bboxes(*outs, metas, rescale=rescale)
        else:
            bbox_list = self.head.get_bboxes(outs, metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def forward_dummy(self, input_data=None, metas=None, rescale=False):
        from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
        metas=[dict(box_type_3d=LiDARInstance3DBoxes)]

        feat = self.extract_feature(input_data, metas)
        outs = self.head(feat)
        if self.head.__class__.__name__ == "Anchor3DHead":
            bbox_list = self.head.get_bboxes(*outs, metas, rescale=rescale)
        else:
            bbox_list = self.head.get_bboxes(outs, metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results