_base_ = [
    './wsood_v2_r50_fpn_1x_dota_le90.py'
]

data_root = '/data/nas/dataset_share/DOTA/split_ss_dota1_0/'
data = dict(
    train=dict(weak_supervised=False),
    val=dict(weak_supervised=False),
    test=dict(weak_supervised=False))
