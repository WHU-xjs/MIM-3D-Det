import cv2
import torch
import numpy as np

import mmcv
from ..builder import PIPELINES
from mmdet3d.datasets.pipelines import RandomFlip3D

@PIPELINES.register_module(name="GlobalRotScaleTrans", force=True)
class GlobalRotScaleTransCustom:
    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False,
                 update_img2lidar=False):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height
        self.update_img2lidar = update_img2lidar

    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor
        
    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation'
                and keys in input_dict['bbox3d_fields'] are updated
                in the result dict.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        # if no bbox in input_dict, only rotate points
        if "points" in input_dict.keys() and len(input_dict['bbox3d_fields']) == 0:
            rot_mat_T = input_dict['points'].rotate(noise_rotation)
            input_dict['pcd_rotation'] = rot_mat_T
            input_dict['pcd_rotation_angle'] = noise_rotation
            return

        # rotate points with bboxes
        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                if "points" in input_dict.keys():
                    points, rot_mat_T = input_dict[key].rotate(
                        noise_rotation, input_dict['points'])
                    input_dict['points'] = points
                else:
                    pseudo_points = torch.zeros((1, 3), 
                            dtype=input_dict[key].tensor.dtype,
                            device=input_dict[key].tensor.device)
                    points, rot_mat_T = input_dict[key].rotate(
                        noise_rotation, pseudo_points)
            else:
                rot_mat_T = torch.eye(3,
                            dtype=input_dict[key].tensor.dtype,
                            device=input_dict[key].tensor.device)
                
            input_dict['pcd_rotation'] = rot_mat_T
            input_dict['pcd_rotation_angle'] = noise_rotation
    
    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        if "points" in input_dict.keys():
            points = input_dict['points']
            points.scale(scale)
            if self.shift_height:
                assert 'height' in points.attribute_dims.keys(), \
                    'setting shift_height=True but points have no height attribute'
                points.tensor[:, points.attribute_dims['height']] *= scale
            input_dict['points'] = points

        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)
    
    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans'
                and keys in input_dict['bbox3d_fields'] are updated
                in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T
        input_dict['pcd_trans'] = trans_factor

        if "points" in input_dict.keys():
            input_dict['points'].translate(trans_factor)
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(trans_factor)

    def _rot_scale_trans_map_mask(self, input_dict):
        map_mask = input_dict["map_mask"]
        _, H, W = map_mask.shape
        res_x = input_dict["xbound"][-1]
        res_y = input_dict["ybound"][-1]

        # get rotation matrix
        if "pcd_rotation" in input_dict.keys():
            rotation_matrix = input_dict["pcd_rotation"].numpy()
        else:
            rotation_matrix = np.identity(3, dtype=np.float32)

        # get scale matrix
        scaling_ratio = input_dict["pcd_scale_factor"]
        scaling_matrix = np.array(
            [[scaling_ratio, 0., 0.], [0., scaling_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)

        # get translation matrix
        trans_x = input_dict["pcd_trans"][0] / res_x
        trans_y = input_dict["pcd_trans"][1] / res_y
        translate_matrix = np.array([[1, 0., trans_y], [0., 1, trans_x], [0., 0., 1.]],
                                      dtype=np.float32)

        c_x = -W/2
        c_y = -H/2
        trans2center_matrix = np.array([[1, 0., c_y], [0., 1, c_x], [0., 0., 1.]],
                                      dtype=np.float32)

        # affine transformation
        warp_matrix = (
            np.linalg.inv(trans2center_matrix) @ translate_matrix @ 
            rotation_matrix @ scaling_matrix @ trans2center_matrix)

        map_mask_cv = np.transpose(map_mask, (1, 2, 0)).astype(np.float32)
        map_mask_warp = cv2.warpPerspective(
            map_mask_cv,
            warp_matrix,
            dsize=(W, H),
            borderValue=0).astype(np.uint8)
        input_dict["map_mask"] = np.transpose(map_mask_warp, (2, 0, 1))

    def _rot_scale_trans_vision_bev(self, input_dict):
        aug_transform = np.zeros((len(input_dict["img"]), 4, 4)).astype(np.float32)
        if self.update_img2lidar:
            aug_transform[:, :3, :3] = input_dict['pcd_rotation'].T * input_dict['pcd_scale_factor']
        else:
            aug_transform[:, :3, :3] = np.eye(3).astype(np.float32) * input_dict['pcd_scale_factor']
        aug_transform[:, :3, 3] = input_dict['pcd_trans'].reshape(1,3)
        aug_transform[:, -1, -1] = 1.0

        input_dict['camera2lidar'] = aug_transform @ input_dict['camera2lidar']

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        # update box & points
        self._rot_bbox_points(input_dict)
        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)
        self._trans_bbox_points(input_dict)

        # update map mask
        if "map_mask" in input_dict:
            self._rot_scale_trans_map_mask(input_dict)

        # update camera2lidar_matrix for vision transformer
        if self.update_img2lidar:
            self._rot_scale_trans_vision_bev(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict
    

@PIPELINES.register_module(name="RandomFlip3D", force=True)
class RandomFlip3DCustom(RandomFlip3D):
    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 **kwargs):
        super(RandomFlip3D, self).__init__(
            flip_ratio=flip_ratio_bev_horizontal, **kwargs)
        self.sync_2d = sync_2d
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
                into result dict.
        """
        # convert imgs to numpy due to flip operation, then back to tensor
        # didnt happen before, maybe due to some pakage version variation
        if 'img' in input_dict:
            input_dict['img'] = [np.array(im) for im in input_dict['img']]
        # flip 2D image and its annotations
        super(RandomFlip3D, self).__call__(input_dict)

        if self.sync_2d:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        rotation = np.eye(3)
        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
            self.random_flip_map_mask(input_dict, 'horizontal') # flip map

        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])
            self.random_flip_map_mask(input_dict, 'vertical') # flip map
        
        if "camera2lidar" in input_dict:
            self.update_transform(input_dict)

        # 'img' should be tuple of tensors
        if 'img' in input_dict.keys():
            input_dict['img'] = [torch.from_numpy(im.copy()) for im in input_dict['img']]
        return input_dict

    def random_flip_map_mask(self, input_dict, direction='horizontal'):
        assert direction in ['horizontal', 'vertical']
        if "map_mask" not in input_dict:
            return
        
        map_mask = input_dict["map_mask"]
        map_mask_cv = np.transpose(map_mask, (1, 2, 0)).astype(np.float32)
        if direction == 'horizontal':
            map_mask_flip = mmcv.imflip(map_mask_cv, "horizontal")
        else:
            map_mask_flip = mmcv.imflip(map_mask_cv, "vertical")
        map_mask_flip = np.transpose(map_mask_flip, (2, 0, 1))
        input_dict["map_mask"] = map_mask_flip

    def update_transform(self, input_dict):
        aug_transform = torch.eye(4).float()
        if input_dict['pcd_horizontal_flip']:
            aug_transform[1, 1] = -1
        if input_dict['pcd_vertical_flip']:
            aug_transform[0, 0] = -1
        aug_transform = aug_transform.view(1,4,4).numpy()
        input_dict['camera2lidar'] = aug_transform @ input_dict['camera2lidar']