from typing import Any, Dict

import torch
import torchvision
import numpy as np
from PIL import Image
import copy
import cv2
from mmdet3d.core.bbox import LiDARInstance3DBoxes as LiDBox
from shapely.geometry import MultiPoint, box
from nuscenes.utils.geometry_utils import view_points
from typing import List, Tuple, Union

from mmcv.utils import build_from_cfg
from ..builder import PIPELINES, OBJECTSAMPLERS

@PIPELINES.register_module()
class ImageAug3D:
    def __init__(
        self, 
        augment=True,
        final_dim=[256, 704],
        resize_lim=(-0.06, 0.11), 
        bot_pct_lim=[0.0, 0.0], 
        rot_lim=(-5.4, 5.4),  # degree
        rand_flip=False,
        resize_test=0.04
    ):
        self.augment = augment
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.resize_test = resize_test

    def sample_augmentation(self, results):
        H, W = results["ori_shape"]
        fH, fW = self.final_dim
        if self.augment:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = float(fW) / float(W)
            resize += self.resize_test
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(
        self, img, rotation, translation, resize, resize_dims, crop, flip, rotate
    ):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data["img"]
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(new_img)
            transforms.append(transform.numpy())
        data["img"] = new_imgs
        # update the calibration matrices
        data["img_aug_matrix"] = np.array(transforms)
        return data

@PIPELINES.register_module(name="ImageNormalize", force=True)
class ImageNormalizeCustom:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # kept for data transformation
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # tensorize = torchvision.transforms.ToTensor()
        # ori_img = [tensorize(img.copy()) for img in data['img']]
        data["img"] = [self.compose(img) for img in data["img"]]
        data["img_norm_cfg"] = dict(mean=self.mean, std=self.std)
        return data

# was: from mmdet3d.datasets.pipelines(file: transforms_3d) import ObjectSample
@PIPELINES.register_module()
class ObjectSampleImage3D(object):
    """
    Args:
        db_sampler (dict): GT sampling class init config
        sort_dist (Bool): whether to use 3d distance in augmentation,
            if True, near SAMPLEs will appear on top of far ones,
            TODO: GTs are not sorted due to 2d bbox calculation error
    """
    def __init__(self, db_sampler, sort_dist=True, stop_epoch=None):
        self.sampler_cfg = db_sampler
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)
        self.sort_dist = sort_dist
        self.epoch = -1
        self.stop_epoch = stop_epoch
        self.views = 6
        self.imgsize = None
        self.lidar2cams = None
        self.intrinsics = None
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def box_lidar2mvimgs(self, boxes3d):
        """ given boxes in LiDAR coordinates, compute their
            2d bboxes on multi-view images
        """
        mvboxes2d = []
        # one-by-one to retain instance correspondency
        for box3d in boxes3d:
            mvbox2d = dict()
            # project every 3d bbox to each camera view
            for view in range(self.views):
                lid2cam, intrin = self.lidar2cams[view], self.intrinsics[view]
                # LIDAR=0, CAM=1, DEPTH=2
                box_cam = LiDBox(box3d[None, :]).convert_to(1, lid2cam)
                corners3d = box_cam.corners.numpy()[0].transpose()
                # 3d -> 2d below: following nuscenes_converter line 495
                in_front = np.argwhere(corners3d[2, :] > 0).flatten()
                corners3d = corners3d[:, in_front]
                corners2d = (view_points(corners3d, intrin, True).T[:, :2].tolist())
                box2d = post_process_coords(corners2d, self.imgsize)
                if box2d is None: continue
                mvbox2d[view] = box2d
            mvboxes2d.append(mvbox2d)
        return mvboxes2d

    def mvbox2d_to_patch(self, mvboxes2d, mvimg):
        """crop objects by their multi-view 2d bbox on images"""
        mvpatches = []
        for mvbox2d in mvboxes2d:
            # double-nested, len(mvpatches) == num_instances
            mvpatches.append([])
            # len(mvpatches[i]) == views_instance_crossed
            if not mvbox2d: continue
            for view, box2d in mvbox2d.items():
                # box2d is tuple of float, round to int for indexing
                xl, yl, xh, yh = np.array(box2d).round().astype(np.int16)
                mvpatches[-1].append(
                    np.array(mvimg[view])[xl:xh+1, yl:yh+1, :])
                # all gts converted patches are wrong!
                # VISUALIZATION
                # cv2.imshow('patch', np.array(mvimg[view])[xl:xh+1, yl:yh+1, :])
                # cv2.waitKey()
                # cv2.destroyWindow('patch')
        return mvpatches

    # TODO: so far there is no obliteration detection
    # for a simplified(and groundtruth preserved) appoarch, 
    # for each sample, if it fully blocks any instance, abort.
    def sort_by_distance(self, instances):
        # calculate distance in LiDAR coordinates
        # approximation for multi-view images to reduce computation
        boxcenter = instances['gt_bboxes_3d'][:,:3]
        boxcenter[:,2] += 0.5 * instances['gt_bboxes_3d'][:,6]
        # d = x^2 + y^2 + z_c^2, z_c = z + h/2
        distances = np.sum(np.square(boxcenter), axis=1)
        # for sort only so no need to root
        rev_idx = np.argsort(distances * -1)
        # VISUALIZATION
        # print('dist')
        # print(distances)
        # print(rev_idx)
        # for patch in instances['gt_patches']:
        #     print('next inst')
        #     cv2.imshow('patch', np.array(patch[0]))
        #     cv2.waitKey()
        # according to observation, it is very likely that
        # with some extremely large objects in gtdb
        # paste near objects obliterate far objects
        # better gen new gtdb first, check sort later
        for key in instances.keys():
            instances[key] = self._take_by_index(instances[key], rev_idx)
        for key in ['gt_bboxes_3d', 'gt_labels_3d']:
            instances[key] = np.stack(instances[key], axis=0)
        return instances

    def _take_by_index(self, arraylike, index):
        return [arraylike[id] for id in index]

    def paste_img_patches(self, instances, images):
        """
        Args:
            instances (dict): objects to be pasted
                - gt_bboxes_2d: list_instances({int_view: list/array_bbox})
                    e.g. [ {0:[xyxy]}, {1:[xyxy], 4:[xyxy]} ]
                    the 2nd instance crosses 2 views (camera 1,4)
                - gt_patches: list_instances(list_view(array_patch))
                    double-nested, N instances' multi-view patches
                    e.g. [ [(35,107)], [(302, 228), (423, 228)] ]
                    the 2nd instance shape (725, 228) crosses 2 views
            images list(np.ndarray): loaded multi-view images
        Returns:
            images (np.ndarray): augmented images
        """
        bboxes2d, patches = instances['gt_bboxes_2d'], instances['gt_patches']
        # for each instance
        for mvbox2d, mvpatch in zip(bboxes2d, patches):
            # for each existing view (though usually 1)
            for [view, bbox], patch in zip(mvbox2d.items(), mvpatch):
                # bbox is tuple of float, round to int
                xl, yl, xh, yh = np.array(bbox).round().astype(np.int16)
                # avoid datatype error in paste, enable resize
                impatch = Image.fromarray(patch)
                # shape order and values can slightly change for Image
                if impatch.size != (xh-xl+1, yh-yl+1):
                    # some correct bbox will still have +0/+1 difference
                    # bilinear, use 1 for lanczos, 3 for bicubic
                    impatch = impatch.resize((xh-xl+1, yh-yl+1), 2)
                images[view].paste(impatch, (xl, yl, xh+1, yh+1))
        # VISUALIZATION
        # imgsnp = [np.array(images[view]) for view in range(self.views)]
        # imgsnp = np.concatenate([
        #     np.concatenate(imgsnp[:len(imgsnp)//2], axis=1), 
        #     np.concatenate(imgsnp[len(imgsnp)//2:], axis=1)
        # ], axis=0)
        # imgsnp = cv2.resize(imgsnp, (1600, 600))
        # cv2.imshow('pasting', imgsnp)
        # cv2.waitKey()
        # cv2.destroyWindow('pasting')
        return images

    def __call__(self, data):
        """Call function to sample ground truth objects to the data.
        Args:
            data (dict): Result dict from all preivous loadings.
        Returns:
            dict: Results after object sampling augmentation, \
                  'img' ,'gt_bboxes_3d', 'gt_labels_3d' keys are updated
        """
        if self.stop_epoch is not None and self.epoch >= self.stop_epoch:
            return data

        # LoadMultiViewImageFromFiles, list(np.ndarray)
        images = data['img']
        self.views = len(images)
        # LoadMapMask
        map_mask = data.get('map_mask', None)
        # LoadAnnotationsMap2D
        gt_bboxes_3d_box = data["gt_bboxes_3d"] # LiDARInstance3DBoxes (N, 9)
        gt_bboxes_3d = gt_bboxes_3d_box.tensor.numpy()
        gt_labels_3d = data["gt_labels_3d"] # ndarray (N,)

        self.imgsize = data['ori_shape'] # original (W,H) for transformation
        self.lidar2cams = data['lidar2camera'] # ndarray (6, 4, 4) for 6 cameras
        self.intrinsics = data['camera_intrinsics'] # ndarray (6, 4, 4)

        # db_sampler sample and filter pseudo-gt annotations
        gts = {
            "gt_bboxes_3d": gt_bboxes_3d, 
            "gt_labels_3d": gt_labels_3d,
            "map_mask": map_mask
        }
        # somehow sample_all affect gts
        samples = self.db_sampler.sample_all(copy.deepcopy(gts))
        # no sample satisfying requirements
        if samples is None: return data

        # augment: paste samples onto images
        # if self.sort_all:
            # # Merge
            # instances= dict()
            # # a whole np.ndarray
            # for key in ['gt_bboxes_3d', 'gt_labels_3d']:
            #     instances[key] = np.concatenate([gts[key], samples[key]], axis=0)
            # # (double-)nested list
            # for key in ['gt_bboxes_2d', 'gt_patches']:
            #     instances[key] = gts[key] + samples[key]
            # gts['gt_bboxes_2d'] = self.box_lidar2mvimgs(gt_bboxes_3d[:,:7])
            # gts['gt_patches'] = self.mvbox2d_to_patch(gts['gt_bboxes_2d'], images)
            # instances = self.merge_and_sort(gts, samples)
        # objects' 2d box position(including view) remains the same
        # if their 3d box are not moved, REGARDLESS of sample they are in
        if self.sort_dist: # sort_sample
            instances = self.sort_by_distance(samples)
        else:
            instances = samples
        aug_imgs = self.paste_img_patches(instances, images)

        # augment: append samples annotations
        sample_bboxes_3d = samples["gt_bboxes_3d"]
        sample_labels = samples["gt_labels_3d"]
        gt_labels_3d = np.concatenate([gt_labels_3d, sample_labels], axis=0)
        gt_bboxes_3d_box = gt_bboxes_3d_box.new_box(
            np.concatenate([gt_bboxes_3d, sample_bboxes_3d])
        )

        data['img'] = aug_imgs # stack to np.ndarray
        data["gt_bboxes_3d"] = gt_bboxes_3d_box
        data["gt_labels_3d"] = gt_labels_3d.astype(np.long)

        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        # repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str

# from nuscenes_converter
def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords]
        )

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None
