_base_ = [
    './rotated_fcos_sep_angle_r50_fpn_1x_dota_le90.py'
]

# evaluation
evaluation = dict(interval=12, metric='mAP')
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[48, 66])
runner = dict(type='EpochBasedRunner', max_epochs=72)
checkpoint_config = dict(interval=12)