dataset_type = 'NuScenesDatasetMap'
data_root = 'data/nuscenes/'
detect_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
map_classes = [
    'drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area',
    'divider'
]
xbound = [-51.2, 51.2, 0.2]
ybound = [-51.2, 51.2, 0.2]
modality = dict(
    use_camera=True,
    use_lidar=False,
    use_radar=False,
    use_map=True,
    use_external=False)
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ImageAug3D',
        augment=True,
        final_dim=[256, 704],
        resize_lim=(-0.06, 0.11),
        rot_lim=[-5.4, 5.4],
        rand_flip=False,
        resize_test=0.04),
    dict(
        type='ImageNormalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='LoadMapMask',
        data_root='data/nuscenes/',
        xbound=[-51.2, 51.2, 0.2],
        ybound=[-51.2, 51.2, 0.2],
        classes=[
            'drivable_area', 'ped_crossing', 'walkway', 'stop_line',
            'carpark_area', 'divider'
        ]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='CollectFusion',
        input_data_keys=('img', 'map_mask', 'camera2lidar',
                         'camera_intrinsics', 'img_aug_matrix'),
        gt_keys=('gt_bboxes_3d', 'gt_labels_3d'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='ImageAug3D',
        augment=False,
        final_dim=[256, 704],
        resize_test=0.04),
    dict(
        type='ImageNormalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='LoadMapMask',
        data_root='data/nuscenes/',
        xbound=[-51.2, 51.2, 0.2],
        ybound=[-51.2, 51.2, 0.2],
        classes=[
            'drivable_area', 'ped_crossing', 'walkway', 'stop_line',
            'carpark_area', 'divider'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(
        type='CollectFusion',
        input_data_keys=('img', 'map_mask', 'camera2lidar',
                         'camera_intrinsics', 'img_aug_matrix'))
]
eval_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='ImageAug3D',
        augment=False,
        final_dim=[256, 704],
        resize_test=0.04),
    dict(
        type='ImageNormalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='LoadMapMask',
        data_root='data/nuscenes/',
        xbound=[-51.2, 51.2, 0.2],
        ybound=[-51.2, 51.2, 0.2],
        classes=[
            'drivable_area', 'ped_crossing', 'walkway', 'stop_line',
            'carpark_area', 'divider'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(
        type='CollectFusion',
        input_data_keys=('img', 'map_mask', 'camera2lidar',
                         'camera_intrinsics', 'img_aug_matrix'))
]
with_velocity = True
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='NuScenesDatasetMap',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nus_img_infos_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(
                type='ImageAug3D',
                augment=True,
                final_dim=[256, 704],
                resize_lim=(-0.06, 0.11),
                rot_lim=[-5.4, 5.4],
                rand_flip=False,
                resize_test=0.04),
            dict(
                type='ImageNormalize',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='LoadMapMask',
                data_root='data/nuscenes/',
                xbound=[-51.2, 51.2, 0.2],
                ybound=[-51.2, 51.2, 0.2],
                classes=[
                    'drivable_area', 'ped_crossing', 'walkway', 'stop_line',
                    'carpark_area', 'divider'
                ]),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0],
                update_img2lidar=True),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='CollectFusion',
                input_data_keys=('img', 'map_mask', 'camera2lidar',
                                 'camera_intrinsics', 'img_aug_matrix'),
                gt_keys=('gt_bboxes_3d', 'gt_labels_3d'))
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        map_classes=[
            'drivable_area', 'ped_crossing', 'walkway', 'stop_line',
            'carpark_area', 'divider'
        ],
        load_interval=1,
        modality=dict(
            use_camera=True,
            use_lidar=False,
            use_radar=False,
            use_map=True,
            use_external=False),
        test_mode=False,
        with_velocity=True,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        type='NuScenesDatasetMap',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nus_img_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='ImageAug3D',
                augment=False,
                final_dim=[256, 704],
                resize_test=0.04),
            dict(
                type='ImageNormalize',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='LoadMapMask',
                data_root='data/nuscenes/',
                xbound=[-51.2, 51.2, 0.2],
                ybound=[-51.2, 51.2, 0.2],
                classes=[
                    'drivable_area', 'ped_crossing', 'walkway', 'stop_line',
                    'carpark_area', 'divider'
                ]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='CollectFusion',
                input_data_keys=('img', 'map_mask', 'camera2lidar',
                                 'camera_intrinsics', 'img_aug_matrix'))
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        map_classes=[
            'drivable_area', 'ped_crossing', 'walkway', 'stop_line',
            'carpark_area', 'divider'
        ],
        with_velocity=True,
        modality=dict(
            use_camera=True,
            use_lidar=False,
            use_radar=False,
            use_map=True,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        load_interval=1),
    test=dict(
        type='NuScenesDatasetMap',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nus_img_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='ImageAug3D',
                augment=False,
                final_dim=[256, 704],
                resize_test=0.04),
            dict(
                type='ImageNormalize',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='LoadMapMask',
                data_root='data/nuscenes/',
                xbound=[-51.2, 51.2, 0.2],
                ybound=[-51.2, 51.2, 0.2],
                classes=[
                    'drivable_area', 'ped_crossing', 'walkway', 'stop_line',
                    'carpark_area', 'divider'
                ]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='CollectFusion',
                input_data_keys=('img', 'map_mask', 'camera2lidar',
                                 'camera_intrinsics', 'img_aug_matrix'))
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        map_classes=[
            'drivable_area', 'ped_crossing', 'walkway', 'stop_line',
            'carpark_area', 'divider'
        ],
        with_velocity=True,
        modality=dict(
            use_camera=True,
            use_lidar=False,
            use_radar=False,
            use_map=True,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR'))
evaluation = dict(
    interval=6,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(
            type='ImageAug3D',
            augment=False,
            final_dim=[256, 704],
            resize_test=0.04),
        dict(
            type='ImageNormalize',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        dict(
            type='LoadMapMask',
            data_root='data/nuscenes/',
            xbound=[-51.2, 51.2, 0.2],
            ybound=[-51.2, 51.2, 0.2],
            classes=[
                'drivable_area', 'ped_crossing', 'walkway', 'stop_line',
                'carpark_area', 'divider'
            ]),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            with_label=False),
        dict(
            type='CollectFusion',
            input_data_keys=('img', 'map_mask', 'camera2lidar',
                             'camera_intrinsics', 'img_aug_matrix'))
    ])
voxel_size = [0.1, 0.1, 0.2]
grid_size_ = [0.8, 0.8]
detect_range_ = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
image_size_ = [256, 704]
c_map = 64
out_size_factor = 1
model = dict(
    type='BEVFusion',
    encoders=[
        dict(
            type='MultiViewEncoder',
            backbone=dict(
                type='SwinTransformer',
                pretrained=
                'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
                pretrain_img_size=224,
                embed_dims=96,
                patch_size=4,
                window_size=7,
                mlp_ratio=4,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                strides=(4, 2, 2, 2),
                out_indices=(2, 3),
                qkv_bias=True,
                qk_scale=None,
                patch_norm=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                use_abs_pos_embed=False,
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='LN', requires_grad=True),
                pretrain_style='official',
                output_missing_index_as_none=False),
            neck=dict(
                type='FPN_LSS',
                in_channels=1152,
                out_channels=512,
                extra_upsample=None,
                input_feature_index=(0, 1),
                scale_factor=2),
            vision_transform=dict(
                type='LSSTransform',
                image_size=[256, 704],
                in_channels=512,
                out_channels=64,
                feature_size=[16, 44],
                pc_range=[-51.2, -51.2, -10.0, 51.2, 51.2, 10.0],
                pc_grid=[0.8, 0.8, 20],
                depth_step=[1.0, 60.0, 1.0],
                downsample=1),
            stream_name='multiview'),
        dict(
            # type='ResizeMapEncoder', # original map (512,512)
            # out_size=(128, 128), # = img = det_range / grid
            # outofmap_layer=False,
            # stream_name="map"
            type="ConvMapEncoder", 
            conv_group_cfg = dict( 
                type="EdgeResidualGroup",
                input_channels=6,
                out_channels=[32, c_map, c_map],
                mid_channels=[16, 32, c_map],
                strides=[2, 2, 1],
                dilation=3,
                with_se=True, # with SE layer
            ),
            outofmap_layer=True,
            stream_name="map"
        ),
    ],
    fuser = dict(
        # type = "ConcatFuser",
        # type = "AddFuser",
        type = "AttentionFuser",
        in_channels=64+c_map,
        attention_only=True,
        attention_cfg=dict(type="BAM"),
        norm_cfg=dict(type='BN'),
    ),
    decoder = dict(
        type = "BaseDecoder",
        backbone = dict(type='ResNetForBEVDet', numC_input=64+c_map, numC_middle=64),
        # backbone = dict(type='ResNetForBEVDet', numC_input=64),
        neck = dict(type='FPN_LSS', in_channels=64*8+64*2, out_channels=256)
    ),
    head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        grid_size=[1024, 1024, 40],
        voxel_size=[0.1, 0.1, 0.2],
        out_size_factor=8,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]),
    test_cfg=dict(
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        score_threshold=0.1,
        pre_max_size=1000,
        post_max_size=83,
        nms_type=['rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
        nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
        nms_rescale_factor=[
            1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
        ]))
optimizer = dict(type='AdamW', lr=5e-05, weight_decay=0.01, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(5, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=24)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'exp/nus/mim_map_test/mim-tiny-rsz-lay6-attn/'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
load_interval_ = 1
gpu_num = 1
base_lr = 5e-05
gpu_ids = [1]
