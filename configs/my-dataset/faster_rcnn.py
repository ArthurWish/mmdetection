_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/multi_coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='SiameseRPNV2',
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,)))
classes = ('merge',)
data = dict(
    samples_per_gpu=1,  # Batch size of a single GPU
    workers_per_gpu=1,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix='my-dataset/train',
        classes=classes,
        ann_file='my-dataset/train/train.json',
    ),
    val=dict(
        img_prefix='my-dataset/test',
        classes=classes,
        ann_file='my-dataset/test/test.json',
    ),
    test=dict(
        img_prefix='my-dataset/test',
        classes=classes,
        ann_file='my-dataset/test/test.json',
    )
)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)

