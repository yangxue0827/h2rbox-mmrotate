from mmrotate.models.builder import ROTATED_LOSSES, build_loss
import torch


@ROTATED_LOSSES.register_module()
class H2RBoxATSSLoss(torch.nn.Module):
    def __init__(self, center_loss_cfg, shape_loss_cfg, angle_loss_cfg,
                 reduction='mean', loss_weight=1.0):
        super(H2RBoxATSSLoss, self).__init__()
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
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted boxes.
            target (torch.Tensor): Corresponding gt boxes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.

        Returns:
            loss (torch.Tensor)
        """
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
