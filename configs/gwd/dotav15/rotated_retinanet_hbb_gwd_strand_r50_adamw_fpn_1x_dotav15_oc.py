_base_ = ['./rotated_retinanet_hbb_gwd_r50_fpn_1x_dotav15_oc.py']

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='GDLoss',
                       loss_type='gwd',
                       fun='sqrt',
                       tau=2.0,
                       loss_weight=5.0,
                       normalize=False)))

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