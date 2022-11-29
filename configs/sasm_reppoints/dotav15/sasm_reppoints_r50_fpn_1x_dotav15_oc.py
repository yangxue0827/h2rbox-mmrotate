_base_ = ['../../rotated_reppoints/dotav15/rotated_reppoints_r50_fpn_1x_dotav15_oc.py']

model = dict(
    bbox_head=dict(
        type='SAMRepPointsHead',
        loss_bbox_init=dict(type='BCConvexGIoULoss', loss_weight=0.375)),

    # training and testing settings
    train_cfg=dict(
        refine=dict(
            _delete_=True,
            assigner=dict(type='SASAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.4),
        max_per_img=2000))
