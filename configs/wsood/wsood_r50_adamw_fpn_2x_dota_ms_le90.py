_base_ = [
    './wsood_r50_adamw_fpn_1x_dota_le90.py'
]

data_root = '/data/nas2/home/zhanggefan/split_ms_dota1_0/'
data = dict(
    train=dict(ann_file=data_root + 'trainval/annfiles/',
               img_prefix=data_root + 'trainval/images/'),
    val=dict(ann_file=data_root + 'trainval/annfiles/',
             img_prefix=data_root + 'trainval/images/'),
    test=dict(ann_file=data_root + 'test/images/',
              img_prefix=data_root + 'test/images/'))

# evaluation
evaluation = dict(interval=12, metric='mAP')
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=4)
