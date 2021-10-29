_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4.py',
    '../_base_/datasets/coco_with_sub_image_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
sub_images = [
    'default.png',
    'filled.png'
]
model = dict(
    type='SiameseRPN',
    backbone=dict(
        type='ResNet',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1
        ),
        type='TridentRoIHead', num_branch=2, test_branch_idx=1),
    train_cfg=dict(
        rpn_proposal=dict(max_per_img=500),
        rcnn=dict(
            sampler=dict(num=128, pos_fraction=0.5,
                         add_gt_as_proposals=False))),
    sub_images = sub_images
)

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# optimizer
optimizer = dict(lr=0.005)
classes = ('merge',)
data = dict(
    samples_per_gpu=4,  # Batch size of a single GPU
    workers_per_gpu=4,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix='my-dataset/train',
        classes=classes,
        ann_file='my-dataset/train/train.json',
        sub_images=sub_images
    ),
    val=dict(
        img_prefix='my-dataset/test',
        classes=classes,
        ann_file='my-dataset/test/test.json',
        sub_images=sub_images
    ),
    test=dict(
        img_prefix='my-dataset/test',
        classes=classes,
        ann_file='my-dataset/test/test.json',
        sub_images=sub_images
    )
)
