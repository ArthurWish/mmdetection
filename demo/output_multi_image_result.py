# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from re import X
from PIL import Image
from matplotlib import patches
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import os
import json
import numpy as np


def predict_img(img_root, img_root_path, img_suffix):
    """use trained model to predict image
    """
    if img_suffix == 'filled.png':
        config = './configs/my-dataset/siamnet_ga_filled.py'
        checkpoint = './work_dirs/siamnet_anchor_ga/img_filled/latest.pth'
    elif img_suffix == 'default.png':
        config = './configs/my-dataset/siamnet_ga_default.py'
        checkpoint = './work_dirs/siamnet_anchor_ga/img_default/latest.pth'
    else:
        raise "invalid image suffix, please use filled or default"
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device='cuda')
    # img_filled = os.path.join(i, 'filled.png')
    img = os.path.join(img_root_path, img_root, img_suffix)
    # test a single image
    result = inference_detector(model, img)
    return model.show_result(img, result, bbox_color=(0, 0, 255), show=False, score_thr=0.3)


def plot_gt_label(img_root, img_root_path, label_file_path, out_file=None, win_name='', thickness=2):
    """plot img with bbox

    Args:
        img ([type]): img file path
        img_root_name ([type]): img root name
        label_file_path ([type]): the json file
        out_file ([type], optional): [description]. Defaults to None.
        win_name (str, optional): [description]. Defaults to ''.
        thickness (int, optional): [description]. Defaults to 1.
    """
    img = os.path.join(img_root_path, img_root, 'default.png')
    img = plt.imread(img)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    currentAxis = fig.gca()
    bboxes = []
    coords = []
    with open(label_file_path) as f:
        data = json.loads(f.read())
    for j in range(len(data["images"])):
        if data["images"][j]["file_name"] == img_root:
            id = j
            break

    for i in range(len(data["annotations"])):
        if data["images"][id]["id"] == data["annotations"][i]["image_id"]:
            bbox = data["annotations"][i]["bbox"]
            bboxes.append(bbox)
    total_obj = len(bboxes)
    bboxes = np.array(bboxes)

    for i, bbox in enumerate(bboxes):
        bbox_int = bbox.astype(np.int32)
        coords.append([bbox_int[0], bbox_int[1], bbox_int[2], bbox_int[3]])

    for _, coord in enumerate(coords):
        rect = patches.Rectangle((coord[0], coord[1]), coord[2], coord[3],
                                 linewidth=thickness, edgecolor='r', facecolor='none')
        currentAxis.add_patch(rect)
    plt.title(f'total_obj:{total_obj}')
    plt.axis('off')
    if out_file is not None:
        plt.savefig(out_file, pad_inches=0.0, bbox_inches='tight')
    return plt.imread(out_file)


def concat_img(img_list: list, img_root, orientation='horizontal', save_dir='demo/result'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    w1, h1, _ = img_list[0].shape
    img_reshaped = []
    for img in img_list:
        img = cv2.resize(img, (w1, w1), interpolation = cv2.INTER_CUBIC)
        img_reshaped.append(img)
    if orientation == 'horizontal':
        img_concat = np.concatenate((img_reshaped), axis=1)
        cv2.imwrite(os.path.join(save_dir, f'{img_root}.png'), img_concat)

if __name__ == '__main__':

    img_root_path = './my-dataset/test/'
    with open(img_root_path + 'test.json') as f:
        data = json.loads(f.read())
    for i in tqdm(range(len(data["images"]))):
        img_root = data["images"][i]["file_name"]
        img_pred_default = predict_img(img_root, img_root_path, img_suffix='default.png')
        img_pred_filled = predict_img(img_root, img_root_path, img_suffix='filled.png')
        img_oringin = plot_gt_label(img_root, img_root_path, label_file_path=img_root_path +
                                    'test.json', out_file='demo/result.png')
        img_oringin = cv2.cvtColor(img_oringin, cv2.COLOR_RGBA2BGR)
        img_oringin = img_oringin * 255.0
        concat_img([img_pred_default, img_pred_filled, img_oringin], img_root)
