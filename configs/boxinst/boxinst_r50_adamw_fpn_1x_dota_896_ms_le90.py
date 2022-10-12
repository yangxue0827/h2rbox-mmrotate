_base_ = [
    './boxinst_r50_adamw_fpn_1x_dota_960_le90.py'
]

angle_version = 'le90'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(896, 896)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='R2H', version=angle_version),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(896, 896),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data_root = '/data/nas/dataset_share/DOTA/split_ss_dota1_0/'
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

data_root = '/data/nas2/home/zhanggefan/split_ms_dota1_0/'
data = dict(
    train=dict(ann_file=data_root + 'trainval/annfiles/',
               img_prefix=data_root + 'trainval/images/'),
    val=dict(ann_file=data_root + 'trainval/annfiles/',
             img_prefix=data_root + 'trainval/images/'),
    test=dict(ann_file=data_root + 'test/images/',
              img_prefix=data_root + 'test/images/'))

checkpoint_config = dict(interval=6)