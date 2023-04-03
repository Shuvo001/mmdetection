_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn_yolox.py',
    '../../_base_/default_runtime.py'
]
max_iters=50000
# dataset settings
classes =  ('MS7U', 'MP1U', 'MU2U', 'ML9U', 'MV1U', 'ML3U', 'MS1U', 'Other')
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='WResNet',
        in_channels=1,
        first_conv_cfg=None,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        deep_stem=True,
        deep_stem_mode='MultiBranchStemS12X',
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type='BN'),
        ),
    rpn_head=dict(
        type='YOLOXRPNHead',
        in_channels=256,
        strides=[24,48,96,192,384],
        feat_channels=256,
        loss_bbox={'type': 'CIoULoss','eps': 1e-16, 'reduction': 'sum', 'loss_weight': 5.0}),
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
            reg_decoded_bbox=True,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(_delete_=True,type='CIoULoss', loss_weight=1.0),
            ),
        ),
        second_stage_hook=dict(type='FusionFPNHook',in_channels=256),
        drop_blocks={ "dropout":{"type":"DropBlock2D","drop_prob":[0.1,0.1,0.1,0.1,0.1],"block_size":[4,4,3,2,1]},
                "scheduler":{"type":"LinearScheduler","begin_step":5000,"end_step":max_iters-5000}},
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', classes_wise_nms=False, iou_threshold=0.2),
                max_per_img=100,
                ),
            min_bbox_size=128,
                ),
        train_cfg=dict(
            rpn=dict(
            assigner=dict(type='SimOTAAssigner', center_radius=2.5),
            ),
            rcnn=dict(
                sampler=dict(
                type='RandomSampler',
                num=128,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
             ),
        )
)
dataset_type = 'WXMLDataset'
data_root = '/home/wj/ai/mldata1/B7mura/datas/train_s0'
test_data_dir = '/home/wj/ai/mldata1/B7mura/datas/test_s0'
#img_scale = (5120, 8192)  # height, width
#random_resize_scales = [8960, 8704, 8448, 8192, 7936, 7680]
#random_crop_scales = [(5600, 8960), (5440, 8704), (5280, 8448), (5120, 8192), (4960, 7936), (4800, 7680)]
img_scale = (3840, 6144)  # height, width
random_resize_scales = [6720, 6528, 6336, 6144, 5952, 5760]
random_crop_scales = [(4006, 6720), (3892, 6528), (3777, 6336), (3663, 6144), (3548, 5952), (3434, 5760)]
img_fill_val = 255
train_pipeline = [
    dict(type='WMosaic', img_scale=img_scale, pad_val=img_fill_val,prob=0.3,skip_filter=False,two_imgs_directions=['horizontal']),
    dict(type="WRandomCrop",crop_if=["WMosaic"],crop_size=random_crop_scales,name="WRandomCrop1",bbox_keep_ratio=0.001,try_crop_around_gtbboxes=True),
    dict(type='WRotate',
        prob=0.3,
        img_fill_val=img_fill_val,
        max_rotate_angle=20.0,
        ),
    dict(type='WTranslate',
        prob=0.3,
        max_translate_offset=200,
        img_fill_val=(img_fill_val,),
        ),
    dict(
        type='WMixUpWithMask',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        prob=0.3,
        pad_val=img_fill_val,skip_filter=False),
    dict(type='WResize', img_scale=random_resize_scales,multiscale_mode=True),
    #dict(type="WRandomCrop",crop_size=random_crop_scales_min,name="WRandomCrop2",bbox_keep_ratio=0.001,try_crop_around_gtbboxes=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=192,img_fill_val=img_fill_val),
    dict(type='WFixData'),
    dict(type='DefaultFormatBundle',img_to_float=False,img_fill_val=img_fill_val),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='WLoadImageFromFile'),
    dict(type='W2Gray'),
    dict(type="WGetImg"),
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        classes=classes,
        img_suffix="jpg",
        ann_file=data_root,
        resample_parameters={"MS1U":8,"ML3U":2,"Other":2},
        pipeline=[
            dict(type='LoadImageFromFile', channel_order="rgb"),
            dict(type='LoadAnnotations', with_bbox=True,with_mask=False),
            dict(type='W2Gray'),
            dict(type='WResize', img_scale=img_scale),
            dict(type="WEncodeImg"),
        ],
        pipeline2=[
            dict(type="WDecodeImg",fmt='gray'),

        ],
        cache_processed_data=True,
        name="b7mura_resample",
    ),
    pipeline=train_pipeline)

samples_per_gpu = 6
data = dict(
    dataloader="mmdet_dataloader",
    data_processor="mmdet_data_processor",
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=10,
    batch_split_nr=2,
    pin_memory=True,
    train= train_dataset,
    val=dict(
        type=dataset_type,
        classes=classes,
        img_suffix="jpg",
        ann_file=data_root,
        data_dirs=test_data_dir,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        img_suffix="jpg",
        ann_file=test_data_dir,
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
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
work_dir="/home/wj/ai/mldata1/B11ACT/workdir/b7mura_faster_yoloxv2_huges_redbcbn"
load_from='/home/wj/ai/work/mmdetection/weights/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
#load_from = '/home/wj/ai/mldata1/B11ACT/workdir/b11act_mask_huge_fp16/weights/checkpoint.pth'
#load_from = '/home/wj/ai/mldata1/B11ACT/workdir/b11act_mask_huge_fp16/weights/checkpoint1.pth'
finetune_model=True
names_not2train = ["backbone"]
names_2train = ["backbone.conv1","backbone.bn1","backbone.stem"]

