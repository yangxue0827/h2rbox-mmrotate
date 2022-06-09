_base_ = [
    '../configs/_base_/datasets/dotav1.py',
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]
angle_version = 'le90'

# model settings
model = dict(
    type='WSL',
    crop_size=(896, 896),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='WSLHead',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        reassigner='one2one',
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_bbox_aug=dict(
            type='WSLRotationLoss',
            loss_weight=0.5,
            center_loss_cfg=dict(type='L1Loss', loss_weight=0.0),
            shape_loss_cfg=dict(type='IoULoss', loss_weight=1.0),
            angle_loss_cfg=dict(type='L1Loss', loss_weight=1.0)),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='FilterNoCenterObj', img_scale=(1100, 1100), crop_size=(896, 896)),
    dict(type='RResize', img_scale=(1100, 1100)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1100, 1100),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=64),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

# data_root = '/data/dataset_share/DOTA/split_ss_dota_wsl/'
data_root = '/data/home/zhanggefan/split_ss_dota_wsl/'
data = dict(
    train=dict(type='DOTAWSLDataset', pipeline=train_pipeline,
               ann_file=data_root + 'trainval/annfiles/',
               img_prefix=data_root + 'trainval/images/',
               version=angle_version),
    val=dict(type='DOTAWSLDataset', pipeline=test_pipeline,
             ann_file=data_root + 'trainval/annfiles/',
             img_prefix=data_root + 'trainval/images/',
             version=angle_version, rect_classes=[9, 11]),
    test=dict(type='DOTAWSLDataset', pipeline=test_pipeline,
              ann_file=data_root + 'test/images/',
              img_prefix=data_root + 'test/images/',
              version=angle_version, rect_classes=[9, 11]))

custom_imports = dict(
    imports=['wsl'],
    allow_failed_imports=False)

log_config = dict(interval=50)
checkpoint_config = dict(interval=6)
optimizer = dict(lr=0.0005)
