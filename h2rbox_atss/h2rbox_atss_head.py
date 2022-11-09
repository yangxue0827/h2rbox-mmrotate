import torch
from mmrotate.models.builder import ROTATED_HEADS, build_loss
from mmrotate.models.dense_heads.rotated_atss_head import RotatedATSSHead
import math
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean, images_to_levels, unmap
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
class H2RBoxATSSHead(RotatedATSSHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 # reassigner='many2one',
                 # assign_vis=False,
                 weak_supervised=True,
                 rotation_agnostic_classes=None,
                 rect_classes=None,
                 angle_version='le135',
                 loss_bbox_aug=dict(type='IoULoss', loss_weight=1.0),
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):

        super(RotatedATSSHead, self).__init__(
            num_classes, in_channels, stacked_convs, conv_cfg, norm_cfg,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

        self.loss_bbox_aug = build_loss(loss_bbox_aug)
        self.rotation_agnostic_classes = rotation_agnostic_classes
        # assert reassigner in ['one2one', 'many2one']
        # self.reassigner = reassigner
        # self.assign_vis = assign_vis
        self.weak_supervised = weak_supervised
        self.rect_classes = rect_classes
        self.angle_version = angle_version

    def forward_aug_single(self, x):

        reg_feat = x
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return bbox_pred,

    def forward_aug(self, feats):
        # outs_aug = []
        # for f in feats:
        #     outs_aug.append(self.forward_aug_single(f))
        # return outs_aug
        return multi_apply(self.forward_aug_single, feats)

    def forward_train(self,
                      x, x_aug, tf,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        # print(len(x_aug), x_aug[0].shape)  # 5 torch.Size([2, 256, 100, 100])
        outs_aug, = self.forward_aug(x_aug)
        # print(len(outs_aug), outs_aug[0].shape)  # 5 torch.Size([2, 5, 100, 100])

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

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (torch.Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (torch.Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            anchors (torch.Tensor): Box reference for each scale level with
                shape (N, num_total_anchors, 5).
            labels (torch.Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (torch.Tensor): Label weights of each anchor with
                shape (N, num_total_anchors)
            bbox_targets (torch.Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 5).
            bbox_weights (torch.Tensor): BBox regression loss weights of each
                anchor with shape (N, num_total_anchors, 5).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            tuple (torch.Tensor):

                - loss_cls (torch.Tensor): cls. loss for each scale level.
                - loss_bbox (torch.Tensor): reg. loss for each scale level.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        if self.weak_supervised:
            loss_bbox = self.loss_bbox(
                obb2xyxy(bbox_pred),
                obb2xyxy(bbox_targets),
                bbox_weights[:, :-1].reshape(-1, 4),
                avg_factor=num_total_samples)
        else:
            loss_bbox = self.loss_bbox(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('outs', 'outs_aug'))
    def loss(self,
             outs, outs_aug, rot,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            outs (list[Tensor]):
                cls_scores (list[Tensor]): Box scores for each scale level
                    Has shape (N, num_anchors * num_classes, H, W)
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level with shape (N, num_anchors * 5, H, W) in View1
            outs_aug (list[Tensor]):
                bbox_preds_aug (list[Tensor]): Box energies / deltas for each scale
                    level with shape (N, num_anchors * 5, H, W) in View2
            rot (int): random rotated angle
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        cls_scores, bbox_preds = outs
        bbox_preds_aug = outs_aug

        assert len(cls_scores) == len(bbox_preds)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)

        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
            for bbox_pred in bbox_preds
        ]

        flatten_anchors = [
            anchors.reshape(-1, 5)
            for anchors in all_anchor_list
        ]
        flatten_labels = [
            labels.reshape(-1, )
            for labels in labels_list
        ]

        flatten_bbox_weights = [
            bbox_weights.reshape(-1, 5)
            for bbox_weights in bbox_weights_list
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_anchors = torch.cat(flatten_anchors)
        flatten_labels = torch.cat(flatten_labels)
        flatten_bbox_weights = torch.cat(flatten_bbox_weights)

        cosa, sina = math.cos(rot), math.sin(rot)
        tf = flatten_cls_scores.new_tensor(
            [[cosa, -sina], [sina, cosa]])

        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_anchors = flatten_anchors[pos_inds]
        pos_labels = flatten_labels[pos_inds]
        pos_bbox_weights = flatten_bbox_weights[pos_inds]

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

        if has_valid_aug:
            pos_inds_aug = torch.cat(pos_inds_aug)
            pos_inds_aug_b = torch.cat(pos_inds_aug_b)
            flatten_bbox_preds_aug = [
                bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
                for bbox_pred in bbox_preds_aug
            ]

            flatten_bbox_preds_aug = torch.cat(flatten_bbox_preds_aug)
            pos_bbox_preds_aug = flatten_bbox_preds_aug[pos_inds_aug]
            pos_anchors_aug = flatten_anchors[pos_inds_aug]

        bbox_coder = self.bbox_coder

        pos_decoded_bbox_preds = bbox_coder.decode(pos_anchors,
                                                   pos_bbox_preds)
        # pos_decoded_target_preds = bbox_coder.decode(
        #     pos_anchors, pos_bbox_targets)

        if has_valid_aug:
            pos_decoded_bbox_preds_aug = bbox_coder.decode(
                pos_anchors_aug, pos_bbox_preds_aug)

            _h, _w = img_metas[0]['crop_size']
            _ctr = tf.new_tensor([[(_w - 1) / 2, (_h - 1) / 2]])

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

            pos_bbox_weights_aug = pos_bbox_weights[pos_inds_aug_v]

            pos_decoded_target_preds_aug = torch.cat(
                [_xy, _wh, pos_angle_targets_aug], dim=-1)

            losses_bbox_aug = self.loss_bbox_aug(
                pos_decoded_bbox_preds_aug,
                pos_decoded_target_preds_aug,
                weight=pos_bbox_weights_aug[:, 0].reshape(-1, ),
                avg_factor=num_total_samples
                )

        else:
            losses_bbox_aug = pos_bbox_preds.sum()

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_bbox_aug=losses_bbox_aug)

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores for a single scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            # angle should not be rescaled
            mlvl_bboxes[:, :4] = mlvl_bboxes[:, :4] / mlvl_bboxes.new_tensor(
                scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms_rotated(
                mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
                cfg.max_per_img)

            if self.rect_classes:
                for id in self.rect_classes:
                    inds = det_labels == id
                    det_bboxes[inds, :-1] = obb2hbb(det_bboxes[inds, :-1], self.angle_version)

            return det_bboxes, det_labels
        else:

            if self.rect_classes:
                for id in self.rect_classes:
                    inds = mlvl_bboxes == id
                    mlvl_bboxes[inds, :-1] = obb2hbb(mlvl_bboxes[inds, :-1], self.angle_version)
            return mlvl_bboxes, mlvl_scores


