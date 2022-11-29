_base_ = [
    './rotated_fcos_sep_angle_r50_adamw_fpn_1x_dotav15_le90.py'
]

# evaluation
evaluation = dict(interval=12, metric='mAP')
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=6)

