import os
import copy
import mmcv
from PIL import Image
import numpy as np
from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.datasets.pipelines import data_augment_utils
from mmdet3d.datasets.pipelines.dbsampler import BatchSampler

@OBJECTSAMPLERS.register_module()
class Map2ImageDataBaseSampler(object):
    """ Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str]): List of classes. Default: None.
    """

    def __init__(
            self,
            info_path,
            data_root,
            rate,
            prepare,
            sample_groups,
            xbound,
            ybound,
            dataset="nuscenes",
            map_filter='none',
            map_classes=None,
            classes=None,
        ):
        super().__init__()
        ## mmdet 3d
        self.data_root = data_root
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}

        db_infos = mmcv.load(info_path)

        # filter database infos
        from mmdet3d.utils import get_root_logger
        logger = get_root_logger()
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        logger.info('After filter database:')
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')

        self.db_infos = db_infos

        # load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)
        # mmdet todo: No group_sampling currently?

        ## Map Init
        self.xbound = xbound
        self.ybound = ybound

        self.dataset = dataset
        self.map_classes = {cls: i for i, cls in enumerate(map_classes)}
        assert map_filter.lower() in ['none', 'onroad', 'refined'], \
            "map_filter must be set to 'none', 'onroad' or 'refined'(case-insensitive)"
        self.map_filter = map_filter.lower()
        if dataset.lower() == "nuscenes":
            assert "drivable_area" in map_classes, "`map_classes` must include 'drivable_area'."
            self.vehicle_names = ["car", "truck", "construction_vehicle", "bus", "trailer"]
        elif dataset.lower() == "lyft":
            self.vehicle_names = ['car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle']
        else:
            raise NotImplementedError("Only surpport NuScenes and Lyft dataset now.")

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
        return new_db_infos

    # TODO: implement visibility filter here

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    def sample_all(self, gt_info):
        """ Sampling all categories of bboxes.
            filter and format intra-class valid samples
        Args:
            gt_info (dict): groundtruths
                - gt_bboxes_3d (np.ndarray)
                - gt_labels_3d (np.ndarray)
                - map_mask
        Returns:
            samples (dict): full info samples
                - gt_labels_3d
                - gt_bboxes_3d
                - gt_bboxes_2d
                - gt_patches
        """
        gt_bboxes = gt_info["gt_bboxes_3d"]
        gt_labels = gt_info["gt_labels_3d"]

        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(
            self.sample_classes, self.sample_max_nums
        ):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(
                max_sample_num - np.sum([n == class_label for n in gt_labels])
            )
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = copy.deepcopy(gt_bboxes)

        for class_name, sampled_num in zip(self.sample_classes, sample_num_per_class):
            if sampled_num > 0:
                gt_info["gt_bboxes_3d"] = avoid_coll_boxes
                if self.map_filter == 'none':
                    sampled_cls = self.sample_class_v2(
                        class_name, sampled_num, gt_info
                    )
                elif self.map_filter == 'onroad':
                    sampled_cls = self.sample_class_onroad(
                        class_name, sampled_num, gt_info
                    )
                elif self.map_filter == 'refined':
                    sampled_cls = self.sample_class_refined(
                        class_name, sampled_num, gt_info
                    )

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s["box3d_lidar"] for s in sampled_cls], axis=0
                        )

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0
                    )

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            # center = sampled_gt_bboxes[:, 0:3]
            sampled_patches = [ # nested, 1 element for 1 view only
                [np.asarray(Image.open(s['patch_path']))] for s in sampled
            ]
            sampled_boxes2d = [s['box2d_camera'] for s in sampled]
            gt_labels = np.array(
                [self.cat2label[s["name"]] for s in sampled], dtype=np.long
            )
            ret = {
                "gt_labels_3d": gt_labels,
                "gt_bboxes_3d": sampled_gt_bboxes,
                'gt_patches': sampled_patches,
                'gt_bboxes_2d': sampled_boxes2d,
            }

        return ret

    def sample_class_v2(self, name, num, gt_info):
        """ Sampling specific categories of bounding boxes.
            return inter-class valid samples from gt_database
        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_info (dict):
                - gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        gt_bboxes = gt_info["gt_bboxes_3d"]
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6]
        )
        
        if len(sampled) == 0:
            return []

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0] :]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]
        )

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples

    def sample_class_onroad(self, name, num, gt_info):
        gt_bboxes = gt_info["gt_bboxes_3d"]
        # gt_labels = gt_info["gt_labels_3d"]
        masks_bev = gt_info["map_mask"]
        assert masks_bev is not None, "Map mask should be loaded first."

        drivable_area_vechiles = (masks_bev[self.map_classes["drivable_area"]] >= 1)
        drivable_area_others = (masks_bev.sum(0) >= 1)

        sampled = []
        num_left = num
        counter = 0
        while len(sampled) < num and counter < 10:
            candidate_sampled = self.sampler_dict[name].sample(num_left)
            for sample in candidate_sampled:
                # construct drivable area
                if sample["name"] in self.vehicle_names:
                    drivable_area = drivable_area_vechiles
                else:
                    drivable_area = drivable_area_others
                
                # check whether box in drivable area
                box_loc2d = copy.deepcopy(sample["box3d_lidar"][:2])
                loc_x = int((box_loc2d[0] - self.xbound[0]) / self.xbound[-1])
                loc_y = int((box_loc2d[1] - self.ybound[0]) / self.ybound[-1])
                if loc_x < 0 or loc_x >= drivable_area.shape[-1] or \
                   loc_y < 0 or loc_y >= drivable_area.shape[0]:
                   continue
                if drivable_area[loc_x, loc_y]:
                    sampled.append(sample)
                    
            num_left = num - len(sampled)
            counter += 1
        
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6]
        )

        if len(sampled) == 0: return []

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0] :]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]
        )

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples

    # TODO: implement refined filtering
    def sample_class_refined(self, name, num, gt_info):
        gt_bboxes = gt_info["gt_bboxes_3d"]
        # A. cancel comment here for class-specified augmentation
        # gt_labels = gt_info["gt_labels_3d"]
        masks_bev = gt_info["map_mask"]
        assert masks_bev is not None, "Map mask should be loaded first."

        # consider acquire and add annotation-visiblity for filtering

        # nuscenes drivable area
        drivable_area_vechiles = (masks_bev[self.map_classes["drivable_area"]] >= 1)
        # any area within map
        drivable_area_others = (masks_bev.sum(0) >= 1)

        sampled = []
        num_left = num
        counter = 0
        # counter controls max tries (regardless of failures)
        # num controls max pasted samples
        #   how to paste desired number of samples proportional to gt samples?
        while len(sampled) < num and counter < 10:
            candidate_sampled = self.sampler_dict[name].sample(num_left)
            for sample in candidate_sampled:
                # construct drivable area
                #   following the code, we can specify:
                #   available_area_class_X = fx(masks_bev)
                #   if 'name' == class_X: available_area = ~_class_X
                if sample["name"] in self.vehicle_names:
                    drivable_area = drivable_area_vechiles
                else:
                    drivable_area = drivable_area_others
                
                # check whether box in drivable area
                #   Question: is this mathmatically correct for all situations?
                #   or how does its visualization look like?
                #   add POSSIBILITY control for avail/unavail area samples
                box_loc2d = copy.deepcopy(sample["box3d_lidar"][:2])
                loc_x = int((box_loc2d[0] - self.xbound[0]) / self.xbound[-1])
                loc_y = int((box_loc2d[1] - self.ybound[0]) / self.ybound[-1])
                if loc_x < 0 or loc_x >= drivable_area.shape[-1] or \
                   loc_y < 0 or loc_y >= drivable_area.shape[0]:
                   continue
                if drivable_area[loc_x, loc_y]:
                    sampled.append(sample)
                    
            num_left = num - len(sampled)
            counter += 1
        
        # all kept and make no changes(?)
        # generate pasted sample annotations
        # filter out collided boxes, get final augmentation samples
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6]
        )

        if len(sampled) == 0: return []
            
        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0] :]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]
        )

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples