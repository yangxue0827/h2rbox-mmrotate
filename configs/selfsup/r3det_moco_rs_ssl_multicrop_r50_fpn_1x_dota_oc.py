_base_ = [
    '../r3det/r3det_r50_fpn_1x_dota_oc.py'
]

model = dict(
    backbone=dict(
        frozen_stages=0,
        zero_init_residual=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/data/nas2/home/zhanggefan/wsl/work_dirs/checkpoint_rs_ssl_multicrop_200.pth'))
    )