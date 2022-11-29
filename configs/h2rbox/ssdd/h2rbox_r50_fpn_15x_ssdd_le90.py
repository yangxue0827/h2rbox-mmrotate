_base_ = [
    '../../_base_/datasets/ssdd.py',
    '../../_base_/schedules/schedule_6x.py',
    '../../_base_/default_runtime.py'
]
angle_version = 'le90'

# model settings
model = dict(
    type='H2RBox',
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
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='H2RBoxHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        reassigner='one2one',
        rect_classes=[],
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
            type='H2RBoxLoss',
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
    # dict(type='FilterNoCenterObject', img_scale=(800, 800), crop_size=(800, 800)),
    # dict(type='RResize', img_scale=(800, 800)),
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
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=64),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data_root = '/data/nas/dataset_share/ssdd_mmrotate/'
data = dict(
    train=dict(type='SARWSOODDataset', pipeline=train_pipeline,
               ann_file=data_root + 'train/labelTxt/',
               img_prefix=data_root + 'train/images/',
               version=angle_version),
    val=dict(type='SARWSOODDataset', pipeline=test_pipeline,
             ann_file=data_root + 'test/inshore/labelTxt/',
             img_prefix=data_root + 'test/inshore/images/',
             version=angle_version),
    test=dict(type='SARWSOODDataset', pipeline=test_pipeline,
              ann_file=data_root + 'test/offshore/labelTxt/',
              img_prefix=data_root + 'test/offshore/images/',
              version=angle_version))

custom_imports = dict(
    imports=['h2rbox'],
    allow_failed_imports=False)

log_config = dict(interval=50)
checkpoint_config = dict(interval=90)
optimizer = dict(lr=0.0005)
evaluation = dict(interval=30, metric='mAP')

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[120, 165])
runner = dict(type='EpochBasedRunner', max_epochs=180)
