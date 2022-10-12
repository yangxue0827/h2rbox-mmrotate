_base_ = [
    './rotated_fcos_sep_angle_r50_adamw_fpn_1x_dota_le90.py'
]

data_root = '/data/nas2/home/zhanggefan/split_ms_dota1_0/'
data = dict(
    train=dict(ann_file=data_root + 'trainval/annfiles/',
               img_prefix=data_root + 'trainval/images/'),
    val=dict(ann_file=data_root + 'trainval/annfiles/',
             img_prefix=data_root + 'trainval/images/'),
    test=dict(ann_file=data_root + 'test/images/',
              img_prefix=data_root + 'test/images/'))

checkpoint_config = dict(interval=6)
