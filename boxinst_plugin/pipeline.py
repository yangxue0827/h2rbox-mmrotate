import torch
from mmrotate.core import obb2xyxy
from mmrotate.datasets.builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class R2H():
    def __init__(self,
                 version):
        self.version = version

    def __call__(self, results):
        for key in results.get('bbox_fields', []):
            results[key] = obb2xyxy(torch.from_numpy(results[key]),
                                    self.version).numpy()
        return results
