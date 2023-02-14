# H2RBox (ICLR 2023)
> [H2RBox: Horizontal Box Annotation is All You Need for Oriented Object Detection](https://arxiv.org/abs/2210.06742)

<!-- [ALGORITHM] -->
## Abstract

<div align=center>
<img src="./configs/h2rbox/pipeline.png" width="800"/>
</div>

Oriented object detection emerges in many applications from aerial images to autonomous driving, while many existing detection benchmarks are annotated with horizontal bounding box only which is also less costive than fine-grained rotated box, leading to a gap between the readily available training corpus and the rising demand for oriented object detection.  This paper proposes a simple yet effective oriented object detection approach called H2RBox merely using horizontal box annotation for weakly-supervised training, which closes the above gap and shows competitive performance even against those trained with rotated boxes.  The cores of our method are weakly- and self-supervised learning, which predicts the angle of the object by learning the consistency of two different views. To our best knowledge, H2RBox is the first horizontal box annotation-based oriented object detector. Compared to an alternative i.e. horizontal box-supervised instance segmentation with our post adaption to oriented object detection, our approach is not susceptible to the prediction quality of mask and can perform more robustly in complex scenes containing a large number of dense objects and outliers. Experimental results show that H2RBox has significant performance and speed advantages over horizontal box-supervised instance segmentation methods, as well as lower memory requirements. While compared to rotated box-supervised oriented object detectors, our method shows very close performance and speed, and even surpasses them in some cases.

## Results and models

DOTA1.0

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | MS  | Batch Size |                                               Configs                                                                   |                                                                                           Download                                                                                           |
|:------------------------:|:-----:|:-----:|:-------:|:--------:|:--------------:|:---:|:----------:|:-----------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ResNet50 (1024,1024,200) | 67.24 | le135 |   1x    |   5.50   |      25.7      |  -  |     2      |             [h2rbox_atss_r50_adamw_fpn_1x_dota_le135](./configs/h2rbox_atss_r50_adamw_fpn_1x_dota_le135.py)             |                                                                                              -                                                                                               |
| ResNet50 (1024,1024,200) | 67.45 | le90  |   1x    |   7.02   |      28.5      |  -  |     2      |                   [h2rbox_r50_adamw_fpn_1x_dota_le90](./configs/h2rbox_r50_adamw_fpn_1x_dota_le90.py)                   | [model](https://drive.google.com/file/d/1pRvlHzeTc71HZQBGdlkjFmeK2RzwC9hS/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1KQ1mtOdAswArm8YGkhXy88LvnOUDIBha/view?usp=sharing) |
| ResNet50 (1024,1024,200) | 70.77 | le90  |   3x    |   7.02   |      28.5      |  -  |     2      |                   [h2rbox_r50_adamw_fpn_3x_dota_le90](./configs/h2rbox_r50_adamw_fpn_3x_dota_le90.py)                   | [model](https://drive.google.com/file/d/1WMtye2T_DOyPMPKbABQsbzIffANEjYpo/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1lRXV2-DsvusgE_W7cVoS7l4O30qwAR7L/view?usp=sharing) |
| ResNet50 (1024,1024,200) | 74.53 | le90  |   1x    |   8.58   |       -        |  √  |     2      |                [h2rbox_r50_adamw_fpn_1x_dota_ms_le90](./configs/h2rbox_r50_adamw_fpn_1x_dota_ms_le90.py)                | [model](https://drive.google.com/file/d/1eY3emcHLs8B0xSU2L3jk0nEcsikSN-vJ/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1qBLvX94qra6UZFncsb7UDYPgC6nssvQd/view?usp=sharing) |

DOTA1.5

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | MS  | Batch Size |                                              Configs                                              | Download |
|:------------------------:|:-----:|:-----:|:-------:|:--------:|:--------------:|:---:|:----------:|:-------------------------------------------------------------------------------------------------:|:--------:|
| ResNet50 (1024,1024,200) | 59.02 | le135 |   1x    |   6.29   |      24.8      |  -  |     2      |   [h2rbox_atss_r50_adamw_fpn_1x_dotav15_le135](./h2rbox_atss_r50_adamw_fpn_1x_dotav15_le135.py)   |    -     |
| ResNet50 (1024,1024,200) | 60.19 | le90  |   1x    |  10.68   |      25.8      |  -  |     2      | [h2rbox_r50_adamw_fpn_1x_dotav15_le90](./configs/dotav15/h2rbox_r50_adamw_fpn_1x_dotav15_le90.py) |    -     |
| ResNet50 (1024,1024,200) | 62.60 | le90  |   1x    |  10.68   |      25.8      |  -  |     2      | [h2rbox_r50_adamw_fpn_3x_dotav15_le90](./configs/dotav15/h2rbox_r50_adamw_fpn_3x_dotav15_le90.py) |    -     |

DOTA2.0

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | MS  | Batch Size |                                            Configs                                             | Download |
|:------------------------:|:-----:|:-----:|:-------:|:--------:|:--------------:|:---:|:----------:|:----------------------------------------------------------------------------------------------:|:--------:|
| ResNet50 (1024,1024,200) | 45.35 | le135 |   1x    |   6.43   |      24.7      |  -  |     2      |  [h2rbox_atss_r50_adamw_fpn_1x_dotav2_le135](./h2rbox_atss_r50_adamw_fpn_1x_dotav2_le135.py)   |    -     |
| ResNet50 (1024,1024,200) | 45.87 | le90  |   1x    |  11.57   |      25.0      |  -  |     2      | [h2rbox_r50_adamw_fpn_1x_dotav2_le90](./configs/dotav2/h2rbox_r50_adamw_fpn_1x_dotav2_le90.py) |    -     |
| ResNet50 (1024,1024,200) | 47.86 | le90  |   1x    |  11.57   |      25.0      |  -  |     2      | [h2rbox_r50_adamw_fpn_3x_dotav2_le90](./configs/dotav2/h2rbox_r50_adamw_fpn_3x_dotav2_le90.py) |    -     |


**Notes:**

- `MS` means multiple scale image split.
- Inf time was tested on a single RTX3090.
- [MMRotate 1.x Implementation for H2RBox](https://github.com/open-mmlab/mmrotate)
- [Jittor Implementation for H2RBox](https://github.com/yangxue0827/h2rbox-jittor)
- [JDet Implementation for H2RBox](https://github.com/Jittor/JDet)

## Get Started

Please refer to the official guide of [MMRotate 0.x](https://github.com/open-mmlab/mmrotate) or [here](./README_en.md).

## Citation
```
@inproceedings{yang2023h2rbox,
  title={H2RBox: Horizontal Box Annotation is All You Need for Oriented Object Detection},
  author={Yang, Xue and Zhang, Gefan and Li, Wentong and Wang, Xuehui and Zhou, Yue and Yan, Junchi},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

```
