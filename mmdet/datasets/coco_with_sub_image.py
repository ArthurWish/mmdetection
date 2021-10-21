import os.path
from .builder import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class CocoDatasetWithSubImage(CocoDataset):
    def __init__(self, sub_images=(), **kwargs):
        self.sub_images = sub_images
        super(CocoDatasetWithSubImage, self).__init__(**kwargs)

    def _parse_ann_info(self, img_info, ann_info):
        ann = super()._parse_ann_info(img_info,ann_info)
        filename_list = [os.path.join(img_info['filename'], sub_image_name) for sub_image_name in self.sub_images]
        img_info['filename'] = filename_list
        seg_map = [x.replace('jpg', 'png') for x in filename_list]
        ann['seg_map']=seg_map
        return ann
