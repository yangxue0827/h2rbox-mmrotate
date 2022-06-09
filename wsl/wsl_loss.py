from mmrotate.models.builder import ROTATED_LOSSES, build_loss
from mmrotate.models.losses import RotatedIoULoss
import torch
from mmdet.models import weight_reduce_loss


@ROTATED_LOSSES.register_module()
class WSLRotationLoss(torch.nn.Module):
    def __init__(self, center_loss_cfg, shape_loss_cfg, angle_loss_cfg,
                 reduction='mean', loss_weight=1.0):
        super(WSLRotationLoss, self).__init__()
        self.center_loss = build_loss(center_loss_cfg)
        self.shape_loss = build_loss(shape_loss_cfg)
        self.angle_loss = build_loss(angle_loss_cfg)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        xy_pred = pred[..., :2]
        xy_target = target[..., :2]
        hbb_pred1 = torch.cat([-pred[..., 2:4], pred[..., 2:4]], dim=-1)
        hbb_pred2 = hbb_pred1[..., [1, 0, 3, 2]]
        hbb_target = torch.cat([-target[..., 2:4], target[..., 2:4]], dim=-1)
        d_a_pred = pred[..., 4] - target[..., 4]

        center_loss = self.center_loss(xy_pred, xy_target,
                                       weight=weight[:, None],
                                       reduction_override=reduction,
                                       avg_factor=avg_factor)
        shape_loss1 = self.shape_loss(hbb_pred1, hbb_target,
                                      weight=weight,
                                      reduction_override=reduction,
                                      avg_factor=avg_factor) + self.angle_loss(
            d_a_pred.sin(), torch.zeros_like(d_a_pred), weight=weight,
            reduction_override=reduction, avg_factor=avg_factor)
        shape_loss2 = self.shape_loss(hbb_pred2, hbb_target,
                                      weight=weight,
                                      reduction_override=reduction,
                                      avg_factor=avg_factor) + self.angle_loss(
            d_a_pred.cos(), torch.zeros_like(d_a_pred), weight=weight,
            reduction_override=reduction, avg_factor=avg_factor)
        loss_bbox = center_loss + torch.min(shape_loss1, shape_loss2)
        return self.loss_weight * loss_bbox


@ROTATED_LOSSES.register_module()
class WSLKFIoULoss(torch.nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(WSLKFIoULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

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

    def forward(self,
                pred,
                target,
                fun='exp',
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        _, Sigma_p = self.xy_wh_r_2_xy_sigma(pred)
        _, Sigma_t = self.xy_wh_r_2_xy_sigma(target)

        Vb_p = 4 * Sigma_p.det().sqrt()
        Vb_t = 4 * Sigma_t.det().sqrt()
        K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
        Sigma = Sigma_p - K.bmm(Sigma_p)
        Vb = 4 * Sigma.det().sqrt()
        Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
        KFIoU = Vb / (Vb_p + Vb_t - Vb + 1e-6)

        if fun == 'ln':
            kf_loss = -torch.log(KFIoU + 1e-6)
        elif fun == 'exp':
            kf_loss = torch.exp(1 - KFIoU) - 1
        else:
            kf_loss = 1 - KFIoU

        loss = kf_loss.clamp(0)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

        return self.loss_weight * loss
