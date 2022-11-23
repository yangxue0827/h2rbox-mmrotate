_base_ = [
    '../../_base_/datasets/dotav2.py', '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

angle_version = 'le135'
model = dict(
    type='H2RBoxATSS',
    crop_size=(1024, 1024),
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
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='H2RBoxATSSHead',
        num_classes=18,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        rect_classes=[9, 11, 16],
        angle_version=angle_version,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        reg_decoded_bbox=True,
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_bbox_aug=dict(
            type='H2RBoxATSSLoss',
            loss_weight=0.4,
            center_loss_cfg=dict(type='L1Loss', loss_weight=0.0),
            shape_loss_cfg=dict(type='IoULoss', loss_weight=1.0),
            angle_loss_cfg=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        assigner=dict(
            type='ATSSObbAssigner',
            topk=9,
            angle_version=angle_version,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
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
    dict(type='FilterNoCenterObject', img_scale=(1024, 1024), crop_size=(1024, 1024)),
    dict(type='RResize', img_scale=(1024, 1024)),
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
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=64),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data_root = '/data/nas/dataset_share/DOTA2.0/split_ss_dota2_0/'
data = dict(
    train=dict(type='DOTAv2WSOODDataset', pipeline=train_pipeline,
               ann_file=data_root + 'trainval/annfiles/',
               img_prefix=data_root + 'trainval/images/',
               version=angle_version),
    val=dict(type='DOTAv2WSOODDataset', pipeline=test_pipeline,
             ann_file=data_root + 'trainval/annfiles/',
             img_prefix=data_root + 'trainval/images/',
             version=angle_version),
    test=dict(type='DOTAv2WSOODDataset', pipeline=test_pipeline,
              ann_file=data_root + 'test/images/',
              img_prefix=data_root + 'test/images/',
              version=angle_version))

custom_imports = dict(
    imports=['h2rbox_atss'],
    allow_failed_imports=False)

log_config = dict(interval=50)
checkpoint_config = dict(interval=6)
optimizer = dict(lr=0.0005)

