_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='GN',num_groups=32),
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[10],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4]),
        bbox_head=dict(
            type='Shared4Conv2FCBBoxHead',
            norm_cfg=dict(type='GN',num_groups=32),
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    second_stage_hook=dict(type='FusionFPNHook',in_channels=256),
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
    name="gds1",
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
        dict(type='WTensorboardLoggerHook',interval=500,
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
#load_from='/home/wj/ai/mldata1/GDS1Crack/mmdet/weights/latest.pth'
finetune_model=True
names_not2train = ["backbone"]
names_2train = None