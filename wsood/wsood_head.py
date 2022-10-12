import torch
from mmrotate.models.builder import ROTATED_HEADS, build_loss
from mmrotate.models.dense_heads.rotated_fcos_head import RotatedFCOSHead, INF
import math
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmrotate.core import multiclass_nms_rotated
from mmrotate.core.bbox.transforms import obb2hbb

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
class WSOODHead(RotatedFCOSHead):
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
                 reassigner='many2one',
                 assign_vis=False,
                 weak_supervised=True,
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
                 rotation_agnostic_classes=None,
                 rect_classes=None,
                 angle_version='le90',
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
        super(WSOODHead, self).__init__(num_classes=num_classes,
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
        self.rotation_agnostic_classes = rotation_agnostic_classes
        assert reassigner in ['one2one', 'many2one']
        self.reassigner = reassigner
        self.assign_vis = assign_vis
        self.weak_supervised = weak_supervised
        self.rect_classes=rect_classes
        self.angle_version=angle_version

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

    def _process_rotation_agnostic(self, tensor, cls, dim=4):
        _rot_agnostic_mask = torch.ones_like(tensor)
        for c in self.rotation_agnostic_classes:
            if dim is None:
                _rot_agnostic_mask[cls == c] = 0
            else:
                _rot_agnostic_mask[cls == c, dim] = 0
        return tensor * _rot_agnostic_mask

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
        labels, bbox_targets, angle_targets, gt_idx = self.get_targets(
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
        flatten_gt_idx = torch.cat(gt_idx)
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
        pos_gt_idx = flatten_gt_idx[pos_inds]
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
            pos_inds_aug_b = []
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
                pos_inds_aug_b.append(b[xy_valid_aug])
                offset += num_imgs * h * w

            has_valid_aug = pos_inds_aug_v.any()

            pos_points = flatten_points[pos_inds]
            pos_labels = flatten_labels[pos_inds]
            if has_valid_aug:
                pos_inds_aug = torch.cat(pos_inds_aug)
                pos_inds_aug_b = torch.cat(pos_inds_aug_b)
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

            if self.weak_supervised:
                loss_bbox = self.loss_bbox(  # todo
                    obb2xyxy(pos_decoded_bbox_preds),
                    obb2xyxy(pos_decoded_target_preds),
                    weight=pos_centerness_targets,
                    avg_factor=centerness_denorm)
            else:
                loss_bbox = self.loss_bbox(
                    pos_decoded_bbox_preds,
                    pos_decoded_target_preds,
                    weight=pos_centerness_targets,
                    avg_factor=centerness_denorm)

                if self.seprate_angle:
                    loss_angle = self.loss_angle(
                        pos_angle_preds, pos_angle_targets, avg_factor=num_pos)

            if has_valid_aug:

                pos_decoded_bbox_preds_aug = bbox_coder.decode(
                    pos_points_aug, pos_bbox_preds_aug)

                _h, _w = img_metas[0]['crop_size']
                _ctr = tf.new_tensor([[(_w - 1) / 2, (_h - 1) / 2]])

                if self.reassigner == 'many2one':
                    ctr_offset = (pos_decoded_bbox_preds[:,
                                  :2] - pos_decoded_target_preds[:, :2]).norm(
                        dim=-1)
                    best = torch.empty_like(ctr_offset, dtype=torch.long)
                    for gt_idx in pos_gt_idx.unique():
                        _s = (pos_gt_idx == gt_idx).nonzero(as_tuple=True)[0]
                        _best = ctr_offset[_s].argmin()
                        best[_s] = _s[_best]

                    best_pos_decoded_bbox_preds = pos_decoded_bbox_preds[best]
                    _xy = best_pos_decoded_bbox_preds[pos_inds_aug_v, :2]
                    _wh = best_pos_decoded_bbox_preds[pos_inds_aug_v, 2:4]
                    pos_angle_targets_aug = best_pos_decoded_bbox_preds[
                                            pos_inds_aug_v, 4:] + rot
                else:
                    _xy = pos_decoded_bbox_preds[pos_inds_aug_v, :2]
                    _wh = pos_decoded_bbox_preds[pos_inds_aug_v, 2:4]
                    pos_angle_targets_aug = pos_decoded_bbox_preds[pos_inds_aug_v,
                                            4:] + rot

                _xy = (_xy - _ctr).matmul(tf.T) + _ctr

                if self.rotation_agnostic_classes:
                    pos_labels_aug = pos_labels[pos_inds_aug_v]
                    pos_angle_targets_aug = self._process_rotation_agnostic(
                        pos_angle_targets_aug,
                        pos_labels_aug, dim=None)

                pos_decoded_target_preds_aug = torch.cat(
                    [_xy, _wh, pos_angle_targets_aug], dim=-1)

                pos_centerness_targets_aug = pos_centerness_targets[
                    pos_inds_aug_v]

                centerness_denorm_aug = max(
                    pos_centerness_targets_aug.sum().detach(), 1)

                with torch.no_grad():
                    if self.assign_vis:
                        from mmrotate.core import imshow_det_rbboxes
                        import numpy as np
                        for i in range(num_imgs):
                            pos_decoded_bbox_preds_aug_i = pos_decoded_bbox_preds_aug[pos_inds_aug_b == i]
                            pos_decoded_target_preds_aug_i = pos_decoded_target_preds_aug[pos_inds_aug_b == i]
                            if self.norm_on_bbox:
                                pos_decoded_bbox_preds_aug_i[:, 2:4] *= self.strides[i]
                                pos_decoded_target_preds_aug_i[:, 2:4] *= self.strides[i]
                            box = torch.cat(
                                [pos_decoded_bbox_preds_aug_i,
                                 pos_decoded_target_preds_aug_i],
                                dim=0).detach().cpu().numpy()
                            labels = np.arange(len(box)) // (len(box) // 2)

                            img = img_metas[i]['visualize_imgs'][1]
                            img = img.detach().permute(1, 2, 0)[
                                ..., [2, 1, 0]].cpu().numpy()
                            img = (img * np.array([58.395, 57.12, 57.375]) + np.array(
                                [123.675, 116.28, 103.53])).clip(0, 255).astype(
                                np.uint8)
                            imshow_det_rbboxes(img, box, labels,
                                               bbox_color=[(255, 0, 0),
                                                           (0, 255, 0)],
                                               show=False,
                                               out_file='./assign_vis_one2one_zero/{}'.format(img_metas[i]['filename'].split('/')[-1]))
                # else:
                #     import mmcv
                #     from mmrotate.core import imshow_det_rbboxes
                #     import numpy as np
                #     for i in range(num_imgs):
                #         box = torch.cat(
                #             [pos_decoded_bbox_preds_aug[pos_inds_aug_b == i],
                #              pos_decoded_target_preds_aug[pos_inds_aug_b == i]],
                #             dim=0).detach().cpu().numpy()
                #         labels = np.arange(len(box)) // (len(box) // 2)
                #         img = mmcv.imread(img_metas[i]['filename'])
                #         img = mmcv.imrotate(img, rot.item() / math.pi * 180)
                #         imshow_det_rbboxes(img, box, labels,
                #                            bbox_color=[(255, 0, 0),
                #                                        (0, 255, 0)])

                loss_bbox_aug = self.loss_bbox_aug(
                    pos_decoded_bbox_preds_aug,
                    pos_decoded_target_preds_aug,
                    weight=pos_centerness_targets_aug,
                    avg_factor=centerness_denorm_aug)

                if self.seprate_angle:
                    loss_angle_aug = self.loss_angle_aug(
                        pos_angle_preds_aug, pos_angle_targets_aug,
                        avg_factor=num_pos)
            else:
                loss_bbox_aug = pos_bbox_preds[[]].sum()
                if self.seprate_angle:
                    loss_angle_aug = pos_bbox_preds[[]].sum()

            # if self.seprate_angle:
            #     loss_angle = self.loss_angle(
            #         pos_angle_preds, pos_angle_targets, avg_factor=num_pos)
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

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        (labels_list, bbox_targets_list, angle_targets_list,
         _gt_idx_list) = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]
        num_gts = [len(gt) for gt in gt_bboxes_list]
        gt_idx_list = []
        gt_idx_offset = 0
        for bid, gt_idx in enumerate(_gt_idx_list):
            gt_idx_list.append(
                (gt_idx + gt_idx_offset).split(num_points, 0))
            gt_idx_offset += num_gts[bid]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        concat_lvl_gt_idx = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            gt_idx = torch.cat([gt_idx[i] for gt_idx in gt_idx_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
            concat_lvl_gt_idx.append(gt_idx)
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets, concat_lvl_gt_idx)

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression, classification and angle targets for a single
        image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1)), \
                   gt_labels.new_full((num_points,), -1),

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                    inside_gt_bbox_mask)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]

        return labels, bbox_targets, angle_targets, min_area_inds

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           angle_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            angle_preds (list[Tensor]): Box angle for a single scale level \
                with shape (N, num_points * 1, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 1, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 6), where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the
                6-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, angle_pred, centerness, points in zip(
                cls_scores, bbox_preds, angle_preds, centernesses,
                mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = self.bbox_coder.decode(
                points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        if self.rect_classes:
            for id in self.rect_classes:
                inds = det_labels==id
                det_bboxes[inds, :-1] = obb2hbb(det_bboxes[inds, :-1], self.angle_version)
        return det_bboxes, det_labels
