# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors.single_stage import RotatedSingleStageDetector
from torch.nn.functional import grid_sample
import math

N = 8


def rotate_feats(feats, theta, ri=True):
    rotated_feats = []
    for feat in feats:
        device = feat.device
        n, c, h, w = feat.shape
        cosa, sina = math.cos(theta), math.sin(theta)
        tf = feat.new_tensor([[cosa, -sina], [sina, cosa]])
        x_range = torch.linspace(-1, 1, w, device=device)
        y_range = torch.linspace(-1, 1, h, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        grid = torch.stack([x, y], -1).expand([n, -1, -1, -1])
        grid = grid.reshape(-1, 2).matmul(tf).view(n, h, w, 2)
        # rotate
        rotated_feat = grid_sample(feat, grid, 'bilinear', 'reflection',
                                   align_corners=True)
        # reorder feature group
        if ri:
            rotated_feat = rotated_feat.view(n, c // N, N, h, w)
            rot = theta * N / 2 / math.pi

            rot_last = math.floor(rot)
            rot_next = math.ceil(rot)

            rot_last_idx = [(i + rot_last) % N for i in range(N)]
            rot_last_feats = rotated_feat[:, :, rot_last_idx, :, :]

            if rot_last == rot_next:
                rotated_feat = rot_last_feats
            else:
                rot_next_ratio = rot - rot_last
                rot_last_ratio = rot_next - rot

                rot_next_idx = [(i + rot_next) % N for i in range(N)]
                rot_next_feats = rotated_feat[:, :, rot_next_idx, :, :]
                rotated_feat = rot_last_feats * rot_last_ratio + \
                               rot_next_feats * rot_next_ratio

        rotated_feats.append(rotated_feat.reshape(n, c, h, w))
    return rotated_feats


@ROTATED_DETECTORS.register_module()
class WSLv2(RotatedSingleStageDetector):
    """Implementation of Rotated `FCOS.`__

    __ https://arxiv.org/abs/1904.01355
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 ri=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(WSLv2, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)
        self.ri = ri

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        rot = (torch.rand(1, device=img.device) * 2 - 1) * math.pi
        x = self.extract_feat(img)
        x_aug = rotate_feats(x, rot, ri=self.ri)
        # with torch.no_grad():
        #     (img2,) = rotate_feats([img], rot, False)
        #     x2 = self.extract_feat(img2)
        #
        #     import matplotlib.pyplot as plt
        #     for i in range(len(x)):
        #         plt.subplot((N + 2) // 2, 4, 1)
        #         plt.imshow(
        #             img[0].mean(dim=0).detach().float().cpu().numpy())
        #         plt.subplot((N + 2) // 2, 4, 3)
        #         plt.title(f'{rot.item() * 180 / math.pi}')
        #         plt.imshow(
        #             img2[0].mean(dim=0).detach().float().cpu().numpy())
        #         gf = x_aug[i][0]
        #         gf = gf.reshape(
        #             gf.shape[0] // N, N, *gf.shape[-2:]).permute(1, 0, 2, 3)
        #         gf2 = x2[i][0]
        #         gf2 = gf2.reshape(
        #             gf2.shape[0] // N, N, *gf2.shape[-2:]).permute(1, 0, 2,
        #                                                            3)
        #         for j, gfj in enumerate(gf):
        #             plt.subplot((N + 2) // 2, 4, 5 + (j % 2) + 4 * (j // 2))
        #             plt.imshow(gfj.detach().mean(0).float().cpu().numpy())
        #         for j, gfj in enumerate(gf2):
        #             plt.subplot((N + 2) // 2, 4, 7 + (j % 2) + 4 * (j // 2))
        #             plt.imshow(gfj.detach().mean(0).float().cpu().numpy())
        #         plt.show()

        losses = self.bbox_head.forward_train(x, x_aug, rot, img_metas,
                                              gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)
        return losses