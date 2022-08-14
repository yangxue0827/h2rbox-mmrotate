_base_ = [
    './wsood_v2_r50_adamw_fpn_1x_dota_960_le90.py'
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
runner = dict(type='EpochBasedRunner', max_epochs=37)
checkpoint_config = dict(interval=6)
