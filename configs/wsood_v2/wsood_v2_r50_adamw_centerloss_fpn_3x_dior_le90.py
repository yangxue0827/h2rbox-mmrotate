_base_ = [
    './wsood_v2_r50_adamw_fpn_3x_dior_le90.py'
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

