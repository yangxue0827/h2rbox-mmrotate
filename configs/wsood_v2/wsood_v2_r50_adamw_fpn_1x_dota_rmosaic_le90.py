_base_ = [
    './wsood_v2_r50_adamw_fpn_1x_dota_le90.py'
]

angle_version = 'le90'
img_scale = (1024, 1024)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='RMosaic', img_scale=img_scale, pad_val=114.0),
    # dict(
    #     type='PolyMixUp',
    #     version=angle_version,
    #     bbox_clip_border=False,
    #     img_scale=img_scale,
    #     ratio_range=(0.8, 1.6),
    #     pad_val=114.0),
    # dict(type='FilterNoCenterObject', img_scale=img_scale, crop_size=img_scale),
    dict(type='RResize', img_scale=img_scale),
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

data_root = '/data/nas/dataset_share/DOTA/split_ss_dota1_0/'
train_dataset = dict(
    _delete_=True,
    type='MultiImageMixDataset',
    dataset=dict(
        type='DOTAWSOODDataset',
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        version=angle_version,
        filter_empty_gt=False
    ),
    pipeline=train_pipeline)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    persistent_workers=True,
    train=train_dataset)
