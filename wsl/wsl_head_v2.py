import torch
from mmrotate.models.builder import ROTATED_HEADS, build_loss
from mmrotate.models.dense_heads.rotated_fcos_head import RotatedFCOSHead, INF
import math
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean


def obb2xyxy(rbboxes):
    w = rbboxes[:, 2::5]
    h = rbboxes[:, 3::5]
    a = rbboxes[:, 4::5]
    cosa = torch.cos(a).abs()
    sina = torch.sin(a).abs()
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    dx = rbboxes[..., 0]
    dy = rbboxes[..., 1]
    dw = hbbox_w.reshape(-1)
    dh = hbbox_h.reshape(-1)
    x1 = dx - dw / 2
    y1 = dy - dh / 2
    x2 = dx + dw / 2
    y2 = dy + dh / 2
    return torch.stack((x1, y1, x2, y2), -1)


@ROTATED_HEADS.register_module()
class WSLHeadv2(RotatedFCOSHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 separate_angle=False,
                 scale_angle=True,
                 h_bbox_coder=dict(type='DistancePointBBoxCoder'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_angle=dict(type='L1Loss', loss_weight=1.0),
                 loss_bbox_aug=dict(type='IoULoss', loss_weight=1.0),
                 loss_angle_aug=dict(type='L1Loss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        super(WSLHeadv2, self).__init__(num_classes=num_classes,
                                      in_channels=in_channels,
                                      regress_ranges=regress_ranges,
                                      center_sampling=center_sampling,
                                      center_sample_radius=center_sample_radius,
                                      norm_on_bbox=norm_on_bbox,
                                      centerness_on_reg=centerness_on_reg,
                                      separate_angle=separate_angle,
                                      scale_angle=scale_angle,
                                      h_bbox_coder=h_bbox_coder,
                                      loss_cls=loss_cls,
                                      loss_bbox=loss_bbox,
                                      loss_angle=loss_angle,
                                      loss_centerness=loss_centerness,
                                      norm_cfg=norm_cfg,
                                      init_cfg=init_cfg,
                                      **kwargs)
        self.loss_bbox_aug = build_loss(loss_bbox_aug)
        if self.seprate_angle:
            self.loss_angle_aug = build_loss(loss_angle_aug)

    def forward_aug_single(self, x, scale, stride):
        reg_feat = x
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        angle_pred = self.conv_angle(reg_feat)
        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()
        return bbox_pred, angle_pred

    def forward_aug(self, feats):
        return multi_apply(self.forward_aug_single, feats, self.scales,
                           self.strides)

    def forward_train(self,
                      x, x_aug, tf,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        outs_aug = self.forward_aug(x_aug)
        if gt_labels is None:
            loss_inputs = (outs, outs_aug, tf) + (gt_bboxes, img_metas)
        else:
            loss_inputs = (outs, outs_aug, tf) + (
                gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    @force_fp32(apply_to=('outs', 'outs_aug'))
    def loss(self,
             outs, outs_aug, rot,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        cls_scores, bbox_preds, angle_preds, centernesses = outs
        bbox_preds_aug, angle_preds_aug = outs_aug

        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            cosa, sina = math.cos(rot), math.sin(rot)
            tf = flatten_cls_scores.new_tensor(
                [[cosa, -sina], [sina, cosa]])

            pos_inds_aug = []
            # pos_inds_aug_b = []
            pos_inds_aug_v = torch.empty_like(pos_inds, dtype=torch.bool)
            offset = 0
            for h, w in featmap_sizes:
                level_mask = (offset <= pos_inds).logical_and(
                    pos_inds < offset + num_imgs * h * w)
                pos_ind = pos_inds[level_mask] - offset
                xy = torch.stack((pos_ind % w, (pos_ind // w) % h), dim=-1)
                b = pos_ind // (w * h)
                ctr = tf.new_tensor([[(w - 1) / 2, (h - 1) / 2]])
                xy_aug = ((xy - ctr).matmul(tf.T) + ctr).round().long()
                x_aug = xy_aug[..., 0]
                y_aug = xy_aug[..., 1]
                xy_valid_aug = ((x_aug >= 0) & (x_aug < w) & (y_aug >= 0) & (
                        y_aug < h))
                pos_ind_aug = (b * h + y_aug) * w + x_aug
                pos_inds_aug_v[level_mask] = xy_valid_aug
                pos_inds_aug.append(pos_ind_aug[xy_valid_aug] + offset)
                # pos_inds_aug_b.append(b[xy_valid_aug])
                offset += num_imgs * h * w

            has_valid_aug = pos_inds_aug_v.any()

            pos_points = flatten_points[pos_inds]
            if has_valid_aug:
                pos_inds_aug = torch.cat(pos_inds_aug)
                # pos_inds_aug_b = torch.cat(pos_inds_aug_b)
                flatten_bbox_preds_aug = [
                    bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
                    for bbox_pred in bbox_preds_aug
                ]
                flatten_angle_preds_aug = [
                    angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
                    for angle_pred in angle_preds_aug
                ]
                flatten_bbox_preds_aug = torch.cat(flatten_bbox_preds_aug)
                flatten_angle_preds_aug = torch.cat(flatten_angle_preds_aug)
                pos_bbox_preds_aug = flatten_bbox_preds_aug[pos_inds_aug]
                pos_angle_preds_aug = flatten_angle_preds_aug[pos_inds_aug]
                pos_points_aug = flatten_points[pos_inds_aug]
            if self.seprate_angle:
                bbox_coder = self.h_bbox_coder
            else:
                bbox_coder = self.bbox_coder
                pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
                                           dim=-1)
                pos_bbox_targets = torch.cat(
                    [pos_bbox_targets, pos_angle_targets], dim=-1)
                if has_valid_aug:
                    pos_bbox_preds_aug = torch.cat(
                        [pos_bbox_preds_aug, pos_angle_preds_aug], dim=-1)

            pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                       pos_bbox_preds)
            pos_decoded_target_preds = bbox_coder.decode(
                pos_points, pos_bbox_targets)
            if has_valid_aug:
                pos_decoded_bbox_preds_aug = bbox_coder.decode(
                    pos_points_aug, pos_bbox_preds_aug)
                _h, _w = img_metas[0]['batch_input_shape']
                _ctr = tf.new_tensor([[_w / 2, _h / 2]])
                _xy = pos_decoded_bbox_preds[pos_inds_aug_v, :2]
                _xy = (_xy - _ctr).matmul(tf.T) + _ctr
                _wh = pos_decoded_bbox_preds[pos_inds_aug_v, 2:4]
                pos_angle_targets_aug = pos_decoded_bbox_preds[pos_inds_aug_v,
                                        4:] + rot
                pos_decoded_target_preds_aug = torch.cat(
                    [_xy, _wh, pos_angle_targets_aug], dim=-1)

                pos_centerness_targets_aug = pos_centerness_targets[
                    pos_inds_aug_v]

                centerness_denorm_aug = max(
                    pos_centerness_targets_aug.sum().detach(), 1)

                loss_bbox_aug = self.loss_bbox_aug(
                    pos_decoded_bbox_preds_aug,
                    pos_decoded_target_preds_aug,
                    weight=pos_centerness_targets_aug,
                    avg_factor=centerness_denorm_aug)

                # from mmrotate.core import imshow_det_rbboxes
                # import numpy as np
                # for i in range(num_imgs):
                #     box = torch.cat(
                #         [pos_decoded_bbox_preds_aug[pos_inds_aug_b == i],
                #          pos_decoded_target_preds_aug[pos_inds_aug_b == i]],
                #         dim=0).detach().cpu().numpy()
                #     labels = np.arange(len(box)) // (len(box) // 2)
                #     img = mmcv.imread(img_metas[i]['filename'])
                #     img = mmcv.imrotate(img, rot.item() / math.pi * 180)
                #     imshow_det_rbboxes(img, box, labels,
                #                        bbox_color=[(255, 0, 0),
                #                                    (0, 255, 0)])

                if self.seprate_angle:
                    loss_angle_aug = self.loss_angle_aug(
                        pos_angle_preds_aug, pos_angle_targets_aug,
                        avg_factor=num_pos)
            else:
                loss_bbox_aug = pos_bbox_preds[[]].sum()
                if self.seprate_angle:
                    loss_angle_aug = pos_bbox_preds[[]].sum()

            loss_bbox = self.loss_bbox(  # todo
                obb2xyxy(pos_decoded_bbox_preds),
                obb2xyxy(pos_decoded_target_preds),
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            if self.seprate_angle:
                loss_angle = self.loss_angle(
                    pos_angle_preds, pos_angle_targets, avg_factor=num_pos)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_bbox_aug = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            if self.seprate_angle:
                loss_angle = pos_angle_preds.sum()
                loss_angle_aug = pos_angle_preds.sum()

        if self.seprate_angle:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_angle=loss_angle,
                loss_centerness=loss_centerness,
                loss_bbox_aug=loss_bbox_aug,
                loss_angle_aug=loss_angle_aug)
        else:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_centerness=loss_centerness,
                loss_bbox_aug=loss_bbox_aug)