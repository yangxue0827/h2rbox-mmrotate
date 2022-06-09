_base_ = [
    '../_base_/datasets/hrsc.py',
    '../_base_/schedules/schedule_6x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

# model settings
model = dict(
    type='WSOOD',
    crop_size=(800, 800),
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
        type='WSOODHead',
        num_classes=1,
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
            type='WSOODLoss',
            loss_weight=0.4,
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
    dict(type='RResize_', img_scale=(800, 800)),
    dict(type='FilterNoCenterObject', img_scale=(800, 800), crop_size=(800, 800)),
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
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize_'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=64),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data_root = '/data/nas/dataset_share/HRSC2016/HRSC2016/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='HRSCWSOODDataset',
        classwise=False,
        ann_file=data_root + 'ImageSets/trainval.txt',
        img_prefix=data_root + 'FullDataSet/AllImages/',
        ann_subdir=data_root + 'FullDataSet/Annotations/',
        img_subdir=data_root + 'FullDataSet/AllImages/',
        version=angle_version,
        pipeline=train_pipeline),
    val=dict(
        type='HRSCWSOODDataset',
        classwise=False,
        ann_file=data_root + 'ImageSets/test.txt',
        img_prefix=data_root + 'FullDataSet/AllImages/',
        ann_subdir=data_root + 'FullDataSet/Annotations/',
        img_subdir=data_root + 'FullDataSet/AllImages/',
        version=angle_version,
        pipeline=test_pipeline),
    test=dict(
        type='HRSCWSOODDataset',
        classwise=False,
        ann_file=data_root + 'ImageSets/test.txt',
        img_prefix=data_root + 'FullDataSet/AllImages/',
        ann_subdir=data_root + 'FullDataSet/Annotations/',
        img_subdir=data_root + 'FullDataSet/AllImages/',
        version=angle_version,
        pipeline=test_pipeline))

custom_imports = dict(
    imports=['wsood'],
    allow_failed_imports=False)

# evaluation
evaluation = dict(interval=72, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[192, 264])
runner = dict(type='EpochBasedRunner', max_epochs=288)
checkpoint_config = dict(interval=72)
