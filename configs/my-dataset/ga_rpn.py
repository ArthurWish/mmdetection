_base_ = ['../guided_anchoring/ga_faster_r50_fpn_1x_coco.py']
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
            bbox_coder=dict(target_stds=[0.05, 0.05, 0.1, 0.1]))))
classes = ('merge',)
data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix='my-dataset/old/train',
        classes=classes,
        ann_file='my-dataset/old/train/train.json',
    ),
    val=dict(
        img_prefix='my-dataset/old/test/images',
        classes=classes,
        ann_file='my-dataset/old/test/images/train.json',
    ),
    test=dict(
        img_prefix='my-dataset/test',
        classes=classes,
        ann_file='my-dataset/old/test/images/train.json',
    )
)
