_base_ = '../rpn/rpn_r50_fpn_1x_coco.py'

model = dict(
    backbone=dict(
        type='SiameseNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))
)

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
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
