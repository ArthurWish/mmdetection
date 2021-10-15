import os.path

import numpy as np

from .api_wrappers import COCO
from .builder import DATASETS
from .coco import CocoDataset


class MultiCOCO(COCO):
    def __init__(self, annotation_file=None):
        self.sub_img_names = {}
        super(MultiCOCO, self).__init__(annotation_file=annotation_file)

    def createIndex(self):
        super(MultiCOCO, self).createIndex()
        if 'sub_images' in self.dataset and isinstance(self.dataset['sub_images'], list):
            def dict_mapping(x):
                if 'id' in x and 'file_name' in x:
                    return x['id'], x['file_name']
                else:
                    return x

            self.sub_img_names = dict(
                map(
                    dict_mapping,
                    self.dataset['sub_images']
                )
            )

    def get_sub_img_names(self):
        return self.sub_img_names


@DATASETS.register_module()
class MultiCocoDataset(CocoDataset):
    def __init__(self, **kwargs):
        self.sub_img_names = {}
        super(MultiCocoDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        self.coco = MultiCOCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        self.sub_img_names = self.coco.get_sub_img_names()

        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = [os.path.join(info['file_name'], sub_img_name) for sub_img_id, sub_img_name in
                                self.sub_img_names.items()]
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = [x.replace('jpg', 'png') for x in img_info['filename']]

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
