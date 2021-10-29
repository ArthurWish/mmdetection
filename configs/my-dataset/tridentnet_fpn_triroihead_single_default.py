_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4.py',
    '../_base_/datasets/coco_with_sub_image_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='SiameseRPNV2',
    backbone=dict(
        type='TridentResNet',
        trident_dilations=(1, 2, 3),
        num_branch=3,
        test_branch_idx=1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1
        ),
        type='TridentRoIHead', num_branch=3, test_branch_idx=1),
    train_cfg=dict(
        rpn_proposal=dict(max_per_img=500),
        rcnn=dict(
            sampler=dict(num=128, pos_fraction=0.5,
                         add_gt_as_proposals=True))))

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
sub_images=[
    'default.png'
]
classes = ('merge',)
data = dict(
    samples_per_gpu=3,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
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
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)