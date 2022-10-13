_base_ = [
    './wsood_v2_r50_fpn_1x_dota_le90.py'
]

# model settings
model = dict(
    bbox_head=dict(
        loss_bbox_aug=dict(
            type='WSOODLoss',
            loss_weight=0.4,
            center_loss_cfg=dict(type='L1Loss', loss_weight=0.15),
            shape_loss_cfg=dict(type='IoULoss', loss_weight=1.0),
            angle_loss_cfg=dict(type='L1Loss', loss_weight=1.0)),
        ))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))