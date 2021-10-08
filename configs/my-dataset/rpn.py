_base_ = '../rpn/rpn_r50_fpn_1x_coco.py'

classes = ('merge',)
data = dict(
    train=dict(
        img_prefix='my-dataset/images',
        classes=classes,
        ann_file='my-dataset/images/train.json'
    ),
    val=dict(
        img_prefix='my-dataset/test/images',
        classes=classes,
        ann_file='my-dataset/test/images/train.json'
    ),
    test=dict(
        img_prefix='my-dataset/test/images',
        classes=classes,
        ann_file='my-dataset/test/images/train.json'
    )
)
runner = dict(type=('EpochBasedRunner'), max_epochs=12)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)