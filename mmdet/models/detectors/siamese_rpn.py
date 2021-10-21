import warnings

import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet.core import bbox_mapping
from mmdet.models.detectors.rpn import RPN
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

@DETECTORS.register_module()
class SiameseRPN(RPN):
    def __init__(self,
                backbone,
                neck,
                rpn_head,
                train_cfg,
                test_cfg,
                init_cfg=None):
        super(SiameseRPN, self).__init__(init_cfg)
        self.backbone_default = build_backbone(backbone)
        self.backbone_fill = build_backbone(backbone)
        self.neck = build_neck(neck) if neck is not None else None
        rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        rpn_head.update(train_cfg=rpn_train_cfg)
        rpn_head.update(test_cfg=test_cfg.rpn)
        self.rpn_head = build_head(rpn_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
