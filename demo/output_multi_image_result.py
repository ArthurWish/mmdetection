# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import os


def main(config, checkpoint, img_root_path):
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device='cuda')
    for i in os.listdir(img_root_path):
        if os.path.splitext(i)[1] == '.json':
            continue
        # img_path = os.path.join(i, 'filled.png')
        img_path = os.path.join(i, 'default.png')
        img = img_root_path + img_path
        # test a single image
        result = inference_detector(model, img)
        # show the results
        # show_result_pyplot(model, img, result, score_thr=0.3)
        out_file = f'./result/img_default/{img_path}'
        model.show_result(img, result, out_file=out_file)
# /home/clq/Desktop/cyn-workspace/remote-ui-detection/configs/my-dataset/siamnet_anchor_ga.py

if __name__ == '__main__':
    config = './configs/my-dataset/ga_rpn.py'
    checkpoint = './work_dirs/siamnet_anchor_ga/img_default/epoch_4.pth'
    img_root_path = './my-dataset/test/'

    main(config, checkpoint, img_root_path)
