# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import points_in_polygons
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

from ..builder import ROTATED_BBOX_ASSIGNERS
from ..transforms import obb2poly


@ROTATED_BBOX_ASSIGNERS.register_module()
class ATSSGaussObbAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): Number of bbox selected in each level.
        use_reassign (bool, optional): If true, it is used to reassign samples.
    """

    def __init__(self, topk, use_reassign=False, angle_version='oc', gauss_type='kld'):
        self.topk = topk
        self.use_reassign = use_reassign
        self.angle_version = angle_version
        self.gauss_type = gauss_type

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. compute the mean aspect ratio of all gts, and set
           exp((-mean aspect ratio / 4) * (mean + std) as the iou threshold
        6. select these candidates whose iou are greater than or equal to
           the threshold as positive
        7. limit the positive sample's center in gt

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        num_gt, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]

        overlaps = self.gauss_overlaps(gt_bboxes, bboxes)
        overlaps = overlaps.transpose(1, 0)
        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        # the center of poly
        gt_points = gt_bboxes[:, :2]
        bboxes_points = bboxes[:, :2]

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(
                self.topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        # add
        gt_bboxes = obb2poly(gt_bboxes, self.angle_version)

        gt_bboxes_ratios = self.aspect_ratio(gt_bboxes)
        gt_bboxes_ratios_per_gt = gt_bboxes_ratios.mean(0)
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        if self.use_reassign:
            iou_thr_weight = torch.exp((-1 / 4) * gt_bboxes_ratios_per_gt)
            overlaps_thr_per_gt = overlaps_thr_per_gt * iou_thr_weight

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # inside_flag = torch.full([num_bboxes, num_gt],
        #                          0.).to(gt_bboxes.device).float()
        # pointsJf(bboxes_points, gt_bboxes, inside_flag)
        inside_flag = points_in_polygons(bboxes_points, gt_bboxes)
        is_in_gts = inside_flag[candidate_idxs,
                                torch.arange(num_gt)].to(is_pos.dtype)

        is_pos = is_pos & is_in_gts
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def gauss_overlaps(self, gt_rbboxes, rbboxes, eps=1e-6):
        """Compute overlaps between polygons and points by Kullback-Leibler
        Divergence loss.

        Args:
            gt_rbboxes (torch.Tensor): Ground truth polygons, shape (k, 5).
            points (torch.Tensor): Points to be assigned, shape(n, 5).
            eps (float, optional): Defaults to 1e-6.

        Returns:
            Tensor: Kullback-Leibler Divergence loss.
        """
        rbboxes = rbboxes.reshape(-1, 5)
        gt_rbboxes = gt_rbboxes.reshape(-1, 5)
        if self.gauss_type == 'kld':
            gauss = self.kld_mixture2single(self.xy_wh_r_2_xy_sigma(rbboxes), self.xy_wh_r_2_xy_sigma(gt_rbboxes))
        elif self.gauss_type == 'gwd':
            gauss = self.gwd_mixture2single(self.xy_wh_r_2_xy_sigma(rbboxes), self.xy_wh_r_2_xy_sigma(gt_rbboxes))
        elif self.gauss_type == 'bcd':
            gauss = self.bcd_mixture2single(self.xy_wh_r_2_xy_sigma(rbboxes), self.xy_wh_r_2_xy_sigma(gt_rbboxes))
        else:
            raise NotImplementedError
        gauss = gauss.clamp(min=eps)
        overlaps = 1 / (1 + gauss)

        return overlaps

    def gwd_mixture2single(self, g1, g2):
        """Compute Wasserstein distance between two Gaussian
        distribution.

        Args:
            g1 (dict[str, torch.Tensor]): Gaussian distribution 1.
            g2 (torch.Tensor): Gaussian distribution 2.

        Returns:
            torch.Tensor: Wasserstein distance.
        """

        if g2[0].shape[0] == 0:
            return g2[0].new_zeros((0, 1))

        p_mu, p_var = g1
        assert p_mu.dim() == 2
        assert p_var.dim() == 3
        N, d = p_mu.shape
        p_mu = p_mu.reshape(1, N, d)
        p_var = p_var.reshape(1, N, d, d)

        t_mu, t_var = g2
        K = t_mu.shape[0]
        t_mu = t_mu.unsqueeze(1)  # (K,1,d)
        t_var = t_var.unsqueeze(1)  # (K,1,d,d)

        delta = p_mu - t_mu
        term1 = torch.sum(delta * delta, dim=-1)
        pt_mul_var = p_var.matmul(t_var)
        term2 = torch.diagonal(p_var + t_var, dim1=-2, dim2=-1).sum(dim=-1) - \
                2 * (torch.diagonal(pt_mul_var, dim1=-2, dim2=-1).sum(dim=-1) +
                     2 * pt_mul_var.det().sqrt()).sqrt()

        # distance = (term1 + term2).clamp(1e-7).sqrt()
        # scale = (p_var.det() * t_var.det()).clamp(1e-7).sqrt()
        # scale = 2 * (
        #     scale.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        # distance = distance / scale
        # return distance

        return term1 + term2  # (K,N)

    def kld_mixture2single(self, g1, g2):
        """Compute Kullback-Leibler Divergence between two Gaussian
        distribution.

        Args:
            g1 (dict[str, torch.Tensor]): Gaussian distribution 1.
            g2 (torch.Tensor): Gaussian distribution 2.

        Returns:
            torch.Tensor: Kullback-Leibler Divergence.
        """
        if g2[0].shape[0] == 0:
            return g2[0].new_zeros((0, 1))

        p_mu, p_var = g1
        assert p_mu.dim() == 2
        assert p_var.dim() == 3
        N, d = p_mu.shape
        p_mu = p_mu.reshape(1, N, d)
        p_var = p_var.reshape(1, N, d, d)

        t_mu, t_var = g2
        K = t_mu.shape[0]
        t_mu = t_mu.unsqueeze(1)
        t_var = t_var.unsqueeze(1)

        delta = (p_mu - t_mu).unsqueeze(-1)
        t_inv = torch.inverse(t_var)
        term1 = delta.transpose(-1, -2).matmul(t_inv).matmul(delta)\
            .reshape(K, N)
        term2 = torch.diagonal(t_inv.matmul(p_var), dim1=-2, dim2=-1)\
            .sum(dim=-1) + torch.log(torch.det(t_var) / torch.det(p_var))

        return 0.5 * (term1 + term2) - 1

    def bcd_mixture2single(self, g1, g2):
        """Compute Bhattacharyya Distance between two Gaussian
        distribution.

        Args:
            g1 (dict[str, torch.Tensor]): Gaussian distribution 1.
            g2 (torch.Tensor): Gaussian distribution 2.

        Returns:
            torch.Tensor: Bhattacharyya Distance.
        """
        if g2[0].shape[0] == 0:
            return g2[0].new_zeros((0, 1))

        p_mu, p_var = g1
        assert p_mu.dim() == 2
        assert p_var.dim() == 3
        N, d = p_mu.shape
        p_mu = p_mu.reshape(1, N, d)
        p_var = p_var.reshape(1, N, d, d)

        t_mu, t_var = g2
        K = t_mu.shape[0]
        t_mu = t_mu.unsqueeze(1)  # (K,1,d)
        t_var = t_var.unsqueeze(1)  # (K,1,d,d)

        delta = (p_mu - t_mu).unsqueeze(-1)  # (K,N,d,1)

        var_sum = 0.5 * (p_var + t_var)
        var_sum_inv = torch.inverse(var_sum)

        term1 = torch.log(torch.det(var_sum) / (torch.sqrt(torch.det(t_var.matmul(p_var)))))
        term2 = delta.transpose(-1, -2).matmul(var_sum_inv).matmul(delta).reshape(K, N)
        return 0.5 * term1 + 0.125 * term2  # (K,N)

    def get_horizontal_bboxes(self, gt_rbboxes):
        """get_horizontal_bboxes from polygons.

        Args:
            gt_rbboxes (torch.Tensor): Groundtruth polygons, shape (k, 8).

        Returns:
            gt_rect_bboxes (torch.Tensor): The horizontal bboxes, shape (k, 4).
        """
        gt_xs, gt_ys = gt_rbboxes[:, 0::2], gt_rbboxes[:, 1::2]
        gt_xmin, _ = gt_xs.min(1)
        gt_ymin, _ = gt_ys.min(1)
        gt_xmax, _ = gt_xs.max(1)
        gt_ymax, _ = gt_ys.max(1)
        gt_rect_bboxes = torch.cat([
            gt_xmin[:, None], gt_ymin[:, None], gt_xmax[:, None], gt_ymax[:,
                                                                          None]
        ],
                                   dim=1)
        return gt_rect_bboxes

    def aspect_ratio(self, gt_rbboxes):
        """compute the aspect ratio of all gts.

        Args:
            gt_rbboxes (torch.Tensor): Groundtruth polygons, shape (k, 8).

        Returns:
            ratios (torch.Tensor): The aspect ratio of gt_rbboxes,
            shape (k, 1).
        """

        pt1, pt2, pt3, pt4 = gt_rbboxes[..., :8].chunk(4, 1)

        edge1 = torch.sqrt(
            torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
            torch.pow(pt1[..., 1] - pt2[..., 1], 2))
        edge2 = torch.sqrt(
            torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
            torch.pow(pt2[..., 1] - pt3[..., 1], 2))

        edges = torch.stack([edge1, edge2], dim=1)
        width, _ = torch.max(edges, 1)
        height, _ = torch.min(edges, 1)
        ratios = width / height
        return ratios

    def xy_wh_r_2_xy_sigma(self, xywhr):
        """Convert oriented bounding box to 2-D Gaussian distribution.

        Args:
            xywhr (torch.Tensor): rbboxes with shape (N, 5).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywhr.shape
        assert _shape[-1] == 5
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = xywhr[..., 4]
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma
