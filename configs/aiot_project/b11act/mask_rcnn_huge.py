_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/default_runtime.py'
]
# dataset settings
classes =  ("burnt","puncture","crease","scratch")
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='WResNet',
        in_channels=1,
        first_conv_cfg={'kernel_size':12,'stride':12,'padding':0},
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
            scales=[15],
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
            featmap_strides=[24]),
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
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=28, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[24]),
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
                nms=dict(type='nms', classes_wise_nms=False, iou_threshold=0.33),
                max_per_img=100,
                mask_thr_binary=0.5)),
        train_cfg=dict(
        rcnn=dict(
            mask_size=56,
            ),
        )
)
dataset_type = 'LabelmeDataset'
data_root = '/home/wj/ai/mldata1/B11ACT/datas/labeled_seg'
#img_scale = (5120, 8192)  # height, width
#random_resize_scales = [8960, 8704, 8448, 8192, 7936, 7680]
#random_crop_scales = [(5600, 8960), (5440, 8704), (5280, 8448), (5120, 8192), (4960, 7936), (4800, 7680)]
img_scale = (3840, 6144)  # height, width
random_resize_scales = [6720, 6528, 6336, 6144, 5952, 5760]
random_crop_scales = [(4200, 6720), (4080, 6528), (3960, 6336), (3840, 6144), (3720, 5952), (3600, 5760)]
random_crop_scales_min = [(2100, 3360), (2040, 3264), (1980, 3168), (1920, 3072), (1860, 2976), (1800, 2880)]
train_pipeline = [
    dict(type='WMosaic', img_scale=img_scale, pad_val=114.0,prob=0.3,skip_filter=False,two_imgs_directions=['horizontal']),
    dict(type="WRandomCrop",crop_if=["WMosaic"],crop_size=random_crop_scales,name="WRandomCrop1"),
    dict(type='WRotate',
        prob=0.3,
        max_rotate_angle=20.0,
        ),
    dict(type='WTranslate',
        prob=0.3,
        max_translate_offset=200,
        img_fill_val=(128,),
        ),
    dict(
        type='WMixUpWithMask',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        prob=0.3,
        pad_val=114.0,skip_filter=False),
    dict(type='WResize', img_scale=random_resize_scales,multiscale_mode=True),
    dict(type="WRandomCrop",crop_size=random_crop_scales_min,name="WRandomCrop2"),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=256),
    dict(type='WFixData'),
    dict(type='W2PolygonMask'),
    dict(type='DefaultFormatBundle',img_to_float=False),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        classes=classes,
        img_suffix="bmp",
        ann_file=data_root,
        pipeline=[
            dict(type='LoadImageFromFile', channel_order="rgb"),
            dict(type='LoadAnnotations', with_bbox=True,with_mask=True),
            dict(type='W2Gray'),
            dict(type='WResize', img_scale=img_scale),
        ],
        cache_processed_data=True,
    ),
    pipeline=train_pipeline)

samples_per_gpu = 8
data = dict(
    dataloader="mmdet_dataloader",
    data_processor="mmdet_data_processor_dm1",
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=6,
    batch_split_nr=2,
    pin_memory=True,
    train= train_dataset,
    val=dict(
        type=dataset_type,
        classes=classes,
        img_suffix="bmp",
        ann_file=data_root,
        data_dirs=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        img_suffix="bmp",
        ann_file=data_root,
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
max_iters=50000
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
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
hooks = [
    dict(type='WMMDetModelSwitch', close_iter=-10000,skip_type_keys=('WMixUpWithMask','WRandomCrop2')),
    dict(type='WMMDetModelSwitch', close_iter=-5000,skip_type_keys=('WMosaic', 'WRandomCrop1','WRandomCrop2', 'WMixUpWithMask')),
]
work_dir="/home/wj/ai/mldata1/B11ACT/workdir/b11act_mask_huge"
#load_from='/home/wj/ai/work/mmdetection/weights/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
load_from = '/home/wj/ai/mldata1/B11ACT/workdir/b11act_mask_huge_fp16/weights/checkpoint_11000.pth'
finetune_model=True
names_not2train = ["backbone"]
names_2train = ["backbone.conv1","backbone.bn1"]

