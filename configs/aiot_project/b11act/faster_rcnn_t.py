_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/default_runtime.py'
]
classes =  ("burnt","puncture","crease","scratch")
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
            scales=[3],
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
            num_classes=len(classes),
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
work_dir="/home/wj/ai/mldata1/B11ACT/workdir/b11act_test"
img_scale = (640, 1024)  # height, width
dataset_type = 'WXMLDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
batch_size = 8
train_dataset = dict(
    type='MosaicDetectionDataset',
    data_dirs=['/home/wj/ai/mldata1/B11ACT/datas/labeled'],
    img_suffix=".jpg;;.bmp",
    classes=classes,
    name="b11act",
    batch_size=batch_size,
    img_size= img_scale,
    allow_empty_annotation=True,
    **img_norm_cfg,
)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    persistent_workers=True,
    pin_memory=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        data_dirs='/home/wj/ai/mldata1/B11ACT/datas/labeled',
        img_suffix=".jpg;;.bmp",
        classes=classes),
    test=dict(
        type=dataset_type,
        data_dirs='/home/wj/ai/mldata1/B11ACT/datas/labeled',
        img_suffix=".jpg;;.bmp",
        classes=classes))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

max_iters=50000
# learning policy
lr_config = dict(
    policy='WarmupCosLR',
    warmup_total_iters=1000,
    total_iters=max_iters)

log_config = dict(
    print_interval=10,
    tb_interval=200)
checkpoint_config = dict(
    interval=1000,
)

hooks = [
    dict(
        type='WCloseMosaic',
        close_iter_or_epoch=-5000,
        by_epoch=False),
]
load_from='/home/wj/ai/work/mmdetection/weights/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
#load_from='/home/wj/ai/mldata1/GDS1Crack/mmdet/weights/latest.pth'
finetune_model=True
names_not2train = ["backbone"]
names_2train = None