_base_ = [
    './wsood_r50_adamw_fpn_1x_dota_le90.py'
]

angle_version = 'le90'

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
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=[9, 11],
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

data_root = '/data/nas2/home/zhanggefan/split_ms_dota1_0/'
data = dict(
    train=dict(type='DOTAWSOODDataset', pipeline=train_pipeline,
               ann_file=data_root + 'trainval/annfiles/',
               img_prefix=data_root + 'trainval/images/',
               version=angle_version),
    val=dict(type='DOTAWSOODDataset', pipeline=test_pipeline,
             ann_file=data_root + 'trainval/annfiles/',
             img_prefix=data_root + 'trainval/images/',
             version=angle_version),
    test=dict(type='DOTAWSOODDataset', pipeline=test_pipeline,
              ann_file=data_root + 'test/images/',
              img_prefix=data_root + 'test/images/',
              version=angle_version))

checkpoint_config = dict(interval=2)
