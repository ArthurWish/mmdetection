import warnings

import mmcv
import torch
from mmcv.image import tensor2imgs
from torch import nn

from mmdet.core import bbox_mapping
from mmdet.models.detectors.faster_rcnn import FasterRCNN
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

@DETECTORS.register_module()
class SiameseRPN(FasterRCNN):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 sub_images=()):

        super(SiameseRPN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.num_branch = self.roi_head.num_branch
        self.backbones = nn.ModuleList([build_backbone(backbone) for i in sub_images])
    
    def extract_feat(self, imgs):
        assert isinstance(imgs, tuple) or isinstance(imgs, list)
        feature_list = [self.backbones[i](img) for i,img in enumerate(imgs)]
        x = tuple(torch.cat(x, dim=0) for x in zip(*feature_list)) # x should be [batch*2, c, h, w]
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, **kwargs):
        trident_gt_bboxes = tuple(gt_bboxes * self.num_branch)
        trident_gt_labels = tuple(gt_labels * self.num_branch)
        trident_img_metas = tuple(img_metas * self.num_branch)
        return super(SiameseRPN,
                     self).forward_train(img, trident_img_metas,
                                         trident_gt_bboxes, trident_gt_labels)
