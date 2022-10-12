_base_ = [
    './wsood_v2_r50_adamw_fpn_1x_dior_le90.py'
]

model = dict(
    bbox_head=dict(reassigner='many2one'))
