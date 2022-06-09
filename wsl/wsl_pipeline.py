import numpy as np
from mmrotate.datasets.builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class FilterNoCenterObj:
    def __init__(self,
                 img_scale=(1100, 1100),
                 crop_size=(896, 896)):
        self.img_scale = img_scale
        self.crop_size = crop_size

    def __call__(self, results):
        bboxes = results['gt_bboxes']
        bboxes = bboxes.reshape((-1, 5))
        xy = bboxes[:, :2]
        flag = xy > (self.img_scale[0] - self.crop_size[0]) // 2
        flag = flag & (xy < ((self.img_scale[1] - self.crop_size[1]) // 2 + self.crop_size[1] - 1))
        flag = flag.all(axis=-1)
        if not flag.any():
            return None
        return results
