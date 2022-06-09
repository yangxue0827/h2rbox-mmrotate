import torch
import numpy as np


def obb2hbb_le90(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
    """
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    hbboxes = torch.cat([center - bias, center + bias], dim=-1)
    _x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    _y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    _w = hbboxes[..., 2] - hbboxes[..., 0]
    _h = hbboxes[..., 3] - hbboxes[..., 1]
    _theta = theta.new_zeros(theta.size(0))
    obboxes1 = torch.stack([_x, _y, _w, _h, _theta], dim=-1)
    obboxes2 = torch.stack([_x, _y, _h, _w, _theta - np.pi / 2], dim=-1)
    obboxes = torch.where((_w >= _h)[..., None], obboxes1, obboxes2)
    return obboxes


def obb2hbb_np_le90(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
    """
    # center, w, h, theta = obboxes[..., :2], obboxes[..., 2], obboxes[..., 3], obboxes[..., 4]
    center, w, h, theta = np.split(obboxes, [2, 3, 4], axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    x_bias = np.abs(w / 2 * Cos) + np.abs(h / 2 * Sin)
    y_bias = np.abs(w / 2 * Sin) + np.abs(h / 2 * Cos)
    bias = np.concatenate([x_bias, y_bias], axis=-1)
    hbboxes = np.concatenate([center - bias, center + bias], axis=-1)
    _x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    _y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    _w = hbboxes[..., 2] - hbboxes[..., 0]
    _h = hbboxes[..., 3] - hbboxes[..., 1]
    _theta = np.zeros(theta.shape[0])
    obboxes1 = np.stack([_x, _y, _w, _h, _theta], axis=-1)
    obboxes2 = np.stack([_x, _y, _h, _w, _theta - np.pi / 2], axis=-1)
    obboxes = np.where((_w >= _h)[..., None], obboxes1, obboxes2)
    import cv2
    cv2.imread()
    return obboxes


if __name__ == '__main__':
    a = np.array([[3.1809e+02, 6.8624e+02, 1.9925e+01, 1.9835e+01, -8.0500e-02],
                  [3.1809e+02, 6.8624e+02, 1.9925e+01, 1.9835e+01, -8.0500e-02]])
    b = obb2hbb_np_le90(a)
    print(b)
    a = torch.from_numpy(a)
    print(obb2hbb_le90(a))
