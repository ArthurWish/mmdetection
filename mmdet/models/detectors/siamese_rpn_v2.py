import torch
from torch import nn
from torchvision import transforms

from .faster_rcnn import FasterRCNN
from ..builder import DETECTORS
from PIL import Image
import numpy as np
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
                 init_cfg=None,
                 sub_images=()):

        super(SiameseRPNV2, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.sub_images = sub_images

    def extract_feat(self, imgs):
        assert len(imgs) == len(self.sub_images)
        # feature_list = []
        # img = torch.cat([img[0], img[1]], dim=1)
        # img = imgs[0].detach().cpu().numpy()
        # im = np.squeeze(img, axis=0).transpose(1,2,0)
        # im = Image.fromarray(im, 'RGB')
        # im.save("xxresult.png")
        
        out = self.backbone(imgs)

        # for input_img in img:
        #     feature = self.backbone(input_img)
        #     # List[list[tensor,tensor,tensor,tensor], list[tensor,tensor,tensor,tensor]]
        #     feature_list.append(feature)
        # feature_concat = tuple(torch.cat(x, dim=1) for x in zip(*feature_list))
        # # print(len(img))
        # out = [
        #     nn.Sequential(
        #         nn.Conv2d(feature_concat_channel.size(1), int(
        #             feature_concat_channel.size(1) / len(img)), kernel_size=1),
        #         nn.BatchNorm2d(int(feature_concat_channel.size(1) / len(img))),
        #         nn.ReLU()
        #     ).cuda()(feature_concat_channel)
        #     for feature_concat_channel in feature_concat
        # ] if len(img) > 1 else feature_concat
        if self.with_neck:
            out = self.neck(out)
        return out
    
    def forward_dummy(self, imgs):
        outs = ()
        # backbone
        x = self.extract_feat(imgs)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(imgs[0].device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      imgs,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(imgs)
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

    def forward_test(self, imgs_list, img_metas, **kwargs):
        for var, name in [(imgs_list, 'imgs_list'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs_list)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs_list)}) '
                             f'!= num of image meta ({len(img_metas)})')

        for imgs, img_meta in zip(imgs_list, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(
                    imgs[0].size()[-2:])
        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs_list[0], img_metas[0], **kwargs)
        else:
            assert imgs_list[0][0].size(0) == 1, 'aug test does not support ' \
                                            'inference with batch size ' \
                                            f'{imgs_list[0][0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs_list, img_metas, **kwargs)

    def onnx_export(self, imgs, img_metas):

        img_shape = torch._shape_as_tensor(imgs[0])[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(imgs)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
