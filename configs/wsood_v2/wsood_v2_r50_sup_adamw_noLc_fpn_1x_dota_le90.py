_base_ = [
    './wsood_v2_r50_sup_adamw_fpn_1x_dota_le90.py'
]

model = dict(
    bbox_head=dict(
        loss_bbox_aug=dict(
            loss_weight=0)))
