_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/default_runtime.py'
]
# dataset settings
classes =  ("burnt","puncture","crease","scratch")
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
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
            type='Shared2FCBBoxHead',
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
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            norm_cfg=dict(type='GN',num_groups=32),
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=len(classes),
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
        second_stage_hook=dict(type='FusionFPNHook',in_channels=256),
            test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', classes_wise_nms=False, iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5))
)
dataset_type = 'LabelmeDataset'
data_root = '/home/wj/ai/mldata1/B11ACT/datas/labeled_seg'
img_scale = (640, 1024)  # height, width
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(img_scale[1], img_scale[0]), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_scale[1], img_scale[0]),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
samples_per_gpu = 8
data = dict(
    dataloader="mmdet_dataloader",
    data_processor="mmdet_data_processor",
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2,
    pin_memory=True,
    train=dict(
        type=dataset_type,
        classes=classes,
        img_suffix="bmp",
        ann_file=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        img_suffix="bmp",
        ann_file=data_root,
        data_dirs='/home/wj/ai/mldata1/B11ACT/datas/labeled',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        img_suffix="bmp",
        ann_file=data_root,
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
max_iters=50000
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
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
work_dir="/home/wj/ai/mldata1/B11ACT/workdir/b11act_mask"
load_from='/home/wj/ai/work/mmdetection/weights/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
finetune_model=True
names_not2train = ["backbone"]
names_2train = None

