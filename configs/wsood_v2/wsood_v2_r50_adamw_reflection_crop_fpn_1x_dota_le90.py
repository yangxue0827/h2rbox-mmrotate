_base_ = [
    './wsood_v2_r50_adamw_fpn_1x_dota_le90.py'
]
angle_version = 'le90'

# model settings
model = dict(crop_size=(704, 704))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='FilterNoCenterObject', img_scale=(1024, 1024), crop_size=(704, 704)),
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

data_root = '/data/nas/dataset_share/DOTA/split_ss_dota1_0/'
data = dict(
    train=dict(pipeline=train_pipeline))
