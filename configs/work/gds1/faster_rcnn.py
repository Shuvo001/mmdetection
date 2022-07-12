_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1
            )
        )
    )
work_dir="/home/wj/ai/mldata1/GDS1Crack/mmdet/weights"
img_scale = (1024, 1024)  # height, width
data_root = '/home/wj/ai/mldata/coco/'
dataset_type = 'WXMLDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
batch_size = 16
train_dataset = dict(
    type='MosaicDetectionDataset',
    data_dirs=[("/home/wj/ai/mldata1/GDS1Crack/train/mdata0",3)],
    img_suffix=".jpg;;.bmp",
    category_index={0:"scratch"},
    batch_size=batch_size,
    **img_norm_cfg,
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        data_dirs="/home/wj/ai/mldata1/GDS1Crack/val/ng",
        img_suffix=".jpg;;.bmp",
        classes=["scratch"],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_dirs="/home/wj/ai/mldata1/GDS1Crack/val/ng",
        img_suffix=".jpg;;.bmp",
        classes=["scratch"],
        pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])

# learning policy
lr_config = dict(
    policy='WCosineAnnealing',
    warmup='exp',
    by_epoch=False,
    warmup_ratio=1,
    warmup_iters=1000,
    min_lr=1e-5)

runner = dict(type='WIterBasedRunner', max_iters=50000)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook',interval=10),
        dict(type='WTensorboardLoggerHook',interval=100,
        log_dir="/home/wj/ai/mldata1/GDS1Crack/tmp/gds1_log",
        mean=img_norm_cfg['mean'],std=img_norm_cfg['std'],rgb=True)
    ])
checkpoint_config = dict(
    interval=1000,
)

custom_hooks = [
    dict(
        type='WCloseMosaic',
        close_iter_or_epoch=45000),
]
load_from='/home/wj/ai/work/mmdetection/weights/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
finetune_model=True
names_not2train = ["backbone"]
names_2train = None