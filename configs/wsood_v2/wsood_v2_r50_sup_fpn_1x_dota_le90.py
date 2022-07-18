_base_ = [
    './wsood_v2_r50_fpn_1x_dota_le90.py'
]

data = dict(
    train=dict(weak_supervised=False),
    val=dict(weak_supervised=False),
    test=dict(weak_supervised=False))
