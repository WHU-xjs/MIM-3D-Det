import pickle
import argparse
from os import path as osp

import mmcv
import numpy as np
# IMPORTANT! otherwise no user model in registry
from method.datasets import *
from mmcv import track_iter_progress
from mmcv.ops import roi_align
from pycocotools.coco import COCO

from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet3d.datasets import build_dataset

# same as loading order!
cam_loading = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
# minimum number of pixels for object to be collected
area_thres = dict(
    car=3500, truck=3500, bus=2800, 
    trailer=2800, construction_vehicle=2800,
    motorcycle=2400, bicycle=2400, pedestrian=2400,
    barrier=2000, traffic_cone=2000,
)
area_max_th = 400*700

def _coco_parse_filter(ann_info):
    gt_bboxes = []
    gt_labels = []
    cameras = []
    instances = []
    
    for i, ann in enumerate(ann_info):
        if ann.get("ignore", False):
            continue
        x1, y1, w, h = ann["bbox"]
        if ann["area"] < area_thres[ann['category_name']]:
            continue
        if ann['area'] > area_max_th:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        gt_bboxes.append(bbox)
        # get camera and instance_token from coco json
        cameras.append(cam_loading.index(ann['camera']))
        instances.append(ann['instance'])

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    ann = dict(bboxes=gt_bboxes, labels=gt_labels, \
               instances=instances, cameras=cameras)

    return ann


def crop_image_patch_v2(pos_proposals, pos_assigned_gt_inds, gt_masks):
    import torch
    from torch.nn.modules.utils import _pair

    device = pos_proposals.device
    num_pos = pos_proposals.size(0)
    fake_inds = torch.arange(num_pos, device=device).to(dtype=pos_proposals.dtype)[
        :, None
    ]
    rois = torch.cat([fake_inds, pos_proposals], dim=1)  # Nx5
    mask_size = _pair(28)
    rois = rois.to(device=device)
    gt_masks_th = (
        torch.from_numpy(gt_masks)
        .to(device)
        .index_select(0, pos_assigned_gt_inds)
        .to(dtype=rois.dtype)
    )
    # Use RoIAlign could apparently accelerate the training (~0.1s/iter)
    targets = roi_align(gt_masks_th, rois, mask_size[::-1], 1.0, 0, True).squeeze(1)
    return targets


def crop_image_patch(pos_proposals, gt_masks, pos_assigned_gt_inds, org_img):
    num_pos = pos_proposals.shape[0]
    masks = []
    img_patches = []
    for i in range(num_pos):
        gt_mask = gt_masks[pos_assigned_gt_inds[i]]
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        mask_patch = gt_mask[y1 : y1 + h, x1 : x1 + w]
        masked_img = gt_mask[..., None] * org_img
        img_patch = masked_img[y1 : y1 + h, x1 : x1 + w]

        img_patches.append(img_patch)
        masks.append(mask_patch)
    return img_patches, masks

def crop_image_patch_instance(instances3d, annotations2d, imgs):
    """Given instances3d token, crop corresponding 2d image patches
    no explicit reference, in each annotation group (2d or 3d),
    instances must be in same length and order as bboxes!

    instances3d (list(str)): instances need to be sampled from
    annotations2d (dict): result from _parse_coco_anno_info
        instances (list(str)): instances(token) given with 2d bbox
        cameras (list(int)): which camera(image_id) 2d bbox is on
        bboxes (list([x1,y1,x2,y2])): 2d bboxes provided
    imgs (list(JpegImageFile)): 6 multi-view images of a sample

    Returns
    img_patches (list(np.ndarray, None)): object image patches
    bbox_avail (list([xyxy], False)): xyxy, False if no bbox
    """
    img_patches = []
    bbox_avail = []
    inst2d, bbox2d = annotations2d['instances'], annotations2d['bboxes']
    cams2d = annotations2d['cameras']
    # some instances can appear across images,
    # which is obviously not welcomed for sampling
    inst_across_imgs = [inst for inst in set(inst2d) if inst2d.count(inst) > 1]
    # filter out those instances along with their cams and bboxes
    gts2d = list(zip(inst2d, zip(cams2d, bbox2d)))
    gts2d = [gt2d for gt2d in gts2d if gt2d[0] not in inst_across_imgs]
    gts2d = dict(gts2d)
    for instance in instances3d:
        if instance in gts2d.keys():
            cam, box = gts2d[instance]
            # crop image patch of instance to sample from
            bbox = box.astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            img_patch = np.array(imgs[cam])[y1 : y1 + h, x1 : x1 + w]
            img_patches.append(img_patch)
            bbox_avail.append({cam: bbox})
        # no bbox found for instance to sample from
        else:
            img_patches.append(None)
            bbox_avail.append(None)
    
    return img_patches, bbox_avail

def create_groundtruth_database(
    dataset_class_name,
    data_path,
    info_prefix,
    info_path=None,
    mask_anno_path=None,
    used_classes=None,
    database_save_path=None,
    db_info_save_path=None,
):
    """Given the raw data, generate the ground truth database.

    mask_anno_path (str): defined by coco_file in $ cmd
        path to coco json file with 2d annotations, None by default
        if given, will generate gtdb_data of image patches of objects
        gtdb_info with key 'bbox' and 'patch'(path to data)
        if None, will only contain basic 3d information
    """
    print(f"Create GT Database of {dataset_class_name}")
    dataset_cfg = dict(
        type=dataset_class_name, data_root=data_path, ann_file=info_path,
        with_velocity=True, with_instance=True,
    )
    if dataset_class_name == "NuScenesDatasetMap":
        dataset_cfg.update(
            # no pts filtering for image detection
            use_valid_flag=False,
            pipeline=[
                # use nuscenes only
                dict(type='LoadMultiViewImageFromFiles', to_float32=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                ),
            ],
        )
    elif dataset_class_name == "LyftDataset":
        dataset_cfg.update(
            pipeline=[
                # not sure
                dict(type='LoadMultiViewImageFromFiles', to_float32=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                ),
            ],
        )
    elif dataset_class_name == "WaymoDataset":
        dataset_cfg.update(
            test_mode=False,
            split="training",
            modality=dict(
                use_lidar=False,
                use_depth=False,
                use_lidar_intensity=False,
                use_camera=True,
            ),
            pipeline=[
                # not sure
                dict(type='LoadMultiViewImageFromFiles', to_float32=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                ),
            ],
        )

    dataset = build_dataset(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f"{info_prefix}_gtdb")
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path, f"{info_prefix}_dbinfos_train.pkl")
    with_bbox = mask_anno_path is not None
    mmcv.mkdir_or_exist(database_save_path)
    all_db_infos = dict()
    if with_bbox:
        coco = COCO(osp.join(data_path, mask_anno_path))
        imgIds = coco.getImgIds()
        file2id = dict()
        for i in imgIds:
            # i: image token (not int index)
            # info['file_name']: img path w.r.t. data_path
            # get basename as file2id key
            imgfile = osp.split(coco.loadImgs([i])[0]['file_name'])[-1]
            file2id.update({imgfile: i})

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        input_dict = dataset.get_data_info(j)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        token = example['token'] # folder name, arrange images
        annos = example["ann_info"]
        image_idx = example["sample_idx"]
        # (should) only name mapping and copying, no reordering
        gt_boxes_3d = annos["gt_bboxes_3d"].tensor.numpy()
        instances3d = annos['gt_instances']
        names = annos["gt_names"]
        group_dict = dict()
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        # if error, check /YOURDataset.py<get_ann_info> mask filter
        assert len(instances3d) == gt_boxes_3d.shape[0], \
            f'boxes {gt_boxes_3d.shape}, instances {len(instances3d)}'
        if with_bbox:
            coco_anns = { # for crop_image_patch_instance
                'instances':[],
                'cameras':[],
                'bboxes':[]
            }
            # for each camera view
            for img_path in example['img_filename']:
                img_path = osp.split(img_path)[-1]
                if img_path not in file2id.keys():
                    print(f"skip image {img_path} for empty mask")
                    continue
                img_id = file2id[img_path]
                # Walkaround for coco.loadAnns(coco.getAnnIds(imgIds=img_id))
                # above returns [], error may caused by using tokens
                kins_raw_info = coco.imgToAnns[img_id]
                kins_ann_info = _coco_parse_filter(kins_raw_info)
                # acquire all view annotations
                for coco_key in coco_anns.keys():
                    # each is list, append can cause hash error
                    coco_anns[coco_key].extend(kins_ann_info[coco_key])
            # intend to utilize coco bbox and segmentation mask
            # here coco 2d was generated from 3d bboxes(no mask)
            obj_img_patches, bbox_avail = crop_image_patch_instance(
                instances3d, coco_anns, example['img']
            )

        # 2d cross-image sample filtering can lead to inconsistency
        num_obj = gt_boxes_3d.shape[0]
        if with_bbox: num_obj = len(bbox_avail)

        for i in range(num_obj):
            # object without 2d bbox are not used
            # note 'not np.array' returns array != 'is not None/False'
            if with_bbox and not bbox_avail[i]: continue

            filename = f"{image_idx}_{names[i]}_{i}"
            abs_filepath = osp.join(database_save_path, token, filename)
            rel_filepath = osp.join(f"{info_prefix}_gtdb", token, filename)

            if with_bbox:
                img_patch_path = abs_filepath + ".png"
                mmcv.imwrite(obj_img_patches[i], img_patch_path)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    "name": names[i],
                    "path": rel_filepath,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes_3d[i],
                    "patch_path": img_patch_path,
                    "difficulty": difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                # db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if with_bbox:
                    db_info.update({"box2d_camera": bbox_avail[i]})
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate groundtruth\
                                     database for image based detection')
    parser.add_argument('--dataset', help='Name of the input dataset', 
                        default='NuScenesDatasetMap')
    parser.add_argument('--data-path', help='Path to dataset',
                        default='./data/nuscenes')
    parser.add_argument('--prefix', help='Prefix of info file(pkl)',
                        default='nus_img')
    parser.add_argument('--info-path', help='Path to info file(pkl)',
                        default='./data/nuscenes/nus_img_infos_train.pkl')
    parser.add_argument('--coco-file', help='Optional, relative path of \
                        coco 2d annotations json file w.r.t. data_path')
    parser.add_argument('--gtdb-data', help='Path to SAVE gt_database files')
    parser.add_argument('--gtdb-info', help='File to SAVE gt_database annos')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    create_groundtruth_database(
        args.dataset,
        args.data_path,
        args.prefix,
        info_path = args.info_path,
        mask_anno_path = args.coco_file,
        used_classes=None,
        database_save_path=args.gtdb_data,
        db_info_save_path=args.gtdb_info,
    )