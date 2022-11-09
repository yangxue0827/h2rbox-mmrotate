# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors.single_stage import RotatedSingleStageDetector
from torch.nn.functional import grid_sample
import math
from mmrotate.core import rbbox2result


@ROTATED_DETECTORS.register_module()
class H2RBoxATSS(RotatedSingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 crop_size=(768, 768),
                 padding='reflection',
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(H2RBoxATSS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained, init_cfg)
        self.crop_size = crop_size
        self.padding = padding

    def rotate_crop(self, img, theta=0., size=(768, 768), gt_bboxes=None, padding='reflection'):
        device = img.device
        n, c, h, w = img.shape
        size_h, size_w = size
        crop_h = (h - size_h) // 2
        crop_w = (w - size_w) // 2
        if theta != 0:
            cosa, sina = math.cos(theta), math.sin(theta)
            tf = img.new_tensor([[cosa, -sina], [sina, cosa]], dtype=torch.float)
            x_range = torch.linspace(-1, 1, w, device=device)
            y_range = torch.linspace(-1, 1, h, device=device)
            y, x = torch.meshgrid(y_range, x_range)
            grid = torch.stack([x, y], -1).expand([n, -1, -1, -1])
            grid = grid.reshape(-1, 2).matmul(tf).view(n, h, w, 2)
            # rotate
            img = grid_sample(img, grid, 'bilinear', padding,
                              align_corners=True)
            if gt_bboxes is not None:
                rot_gt_bboxes = []
                for bboxes in gt_bboxes:
                    xy, wh, a = bboxes[..., :2], bboxes[..., 2:4], bboxes[..., [4]]
                    ctr = tf.new_tensor([[w / 2, h / 2]])
                    xy = (xy - ctr).matmul(tf.T) + ctr
                    a = a + theta
                    rot_gt_bboxes.append(torch.cat([xy, wh, a], dim=-1))
                gt_bboxes = rot_gt_bboxes
        img = img[..., crop_h: crop_h + size_h, crop_w:crop_w + size_w]
        if gt_bboxes is None:
            return img
        else:
            crop_gt_bboxes = []
            for bboxes in gt_bboxes:
                xy, wh, a = bboxes[..., :2], bboxes[..., 2:4], bboxes[..., [4]]
                xy = xy - xy.new_tensor([[crop_w, crop_h]])
                crop_gt_bboxes.append(torch.cat([xy, wh, a], dim=-1))
            gt_bboxes = crop_gt_bboxes

            # from mmrotate.core import imshow_det_rbboxes
            # import numpy as np
            # for i, bboxes in enumerate(gt_bboxes):
            #     _img = img[i].detach().permute(1, 2, 0)[
            #         ..., [2, 1, 0]].cpu().numpy()
            #     _img = (_img * np.array([58.395, 57.12, 57.375]) + np.array(
            #         [123.675, 116.28, 103.53])).clip(0, 255).astype(
            #         np.uint8)
            #     imshow_det_rbboxes(_img, bboxes=bboxes.detach().cpu().numpy(),
            #                        labels=np.arange(len(bboxes)))

            return img, gt_bboxes

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        rot = (torch.rand(1, device=img.device) * 2 - 1) * math.pi
        # rot = 0.25 * math.pi
        if self.train_cfg is not None:
            self.crop_size = self.train_cfg.get('crop_size', self.crop_size)
        img1, gt_bboxes = self.rotate_crop(img, 0, self.crop_size, gt_bboxes, self.padding)
        x = self.extract_feat(img1)
        img2 = self.rotate_crop(img, rot, self.crop_size, padding=self.padding)
        x_aug = self.extract_feat(img2)
        for idx, m in enumerate(img_metas):
            m['crop_size'] = self.crop_size
            m['visualize_imgs'] = (img1[idx], img2[idx])
        losses = self.bbox_head.forward_train(x, x_aug, rot, img_metas,
                                              gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)
        return losses
