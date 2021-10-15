import torch
from torch import nn

from .faster_rcnn import FasterRCNN
from ..builder import DETECTORS


@DETECTORS.register_module()
class SiameseRPNV2(FasterRCNN):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):

        super(SiameseRPNV2, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    # def extract_feat(self, img):
    #     img = img[0]
    #     x = self.backbone(img)
    #     if self.with_neck:
    #         x = self.neck(x)
    #     return x

    def extract_feat(self, img):
        assert isinstance(img, tuple) or isinstance(img, list)
        feature_list = []
        for input_img in img:
            feature = self.backbone(input_img)
            # List[list[tensor,tensor,tensor,tensor], list[tensor,tensor,tensor,tensor]]
            feature_list.append(feature)
        feature_concat = tuple(torch.cat(x, dim=1) for x in zip(*feature_list))
        out = [
            nn.Sequential(
                nn.Conv2d(feature_concat_channel.size(1), int(
                    feature_concat_channel.size(1) / len(img)), kernel_size=1),
                nn.BatchNorm2d(int(feature_concat_channel.size(1) / len(img))),
                nn.ReLU()
            ).cuda()(feature_concat_channel)
            for feature_concat_channel in feature_concat
        ] if len(img) > 1 else feature_concat
        if self.with_neck:
            out = self.neck(out)
        return out

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)
        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(
                    img[0].size()[-2:])
        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0][0].size(0) == 1, 'aug test does not support ' \
                                            'inference with batch size ' \
                                            f'{imgs[0][0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)
