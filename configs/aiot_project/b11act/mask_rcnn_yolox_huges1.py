_base_ = [
    './mask_rcnn_yolox_huges.py'
]
# dataset settings
classes =  ("burnt","puncture","crease","scratch")
dataset_type = 'LabelmeDataset'
data_root = '/home/wj/ai/mldata1/B11ACT/datas/labeled_seg'
#img_scale = (5120, 8192)  # height, width
#random_resize_scales = [8960, 8704, 8448, 8192, 7936, 7680]
#random_crop_scales = [(5600, 8960), (5440, 8704), (5280, 8448), (5120, 8192), (4960, 7936), (4800, 7680)]
img_scale = (3840, 6144)  # height, width
random_resize_scales = [6720, 6528, 6336, 6144, 5952, 5760]
random_crop_scales = [(4006, 6720), (3892, 6528), (3777, 6336), (3663, 6144), (3548, 5952), (3434, 5760)]
train_pipeline = [
    dict(type='WMosaic', img_scale=img_scale, pad_val=114.0,prob=0.3,skip_filter=False,two_imgs_directions=['horizontal']),
    dict(type="WRandomCrop",crop_if=["WMosaic"],crop_size=random_crop_scales,name="WRandomCrop1",bbox_keep_ratio=0.001,try_crop_around_gtbboxes=True),
    dict(type='WRotate',
        prob=0.3,
        img_fill_val=0,
        max_rotate_angle=20.0,
        ),
    dict(type='WTranslate',
        prob=0.3,
        max_translate_offset=200,
        img_fill_val=(0,),
        ),
    dict(
        type='WMixUpWithMask',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        prob=0.3,
        pad_val=114.0,skip_filter=False),
    dict(type='WResize', img_scale=random_resize_scales,multiscale_mode=True),
    #dict(type="WRandomCrop",crop_size=random_crop_scales_min,name="WRandomCrop2",bbox_keep_ratio=0.001,try_crop_around_gtbboxes=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=192),
    dict(type='WFixData'),
    dict(type='W2PolygonMask'),
    dict(type='DefaultFormatBundle',img_to_float=False),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
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

samples_per_gpu = 6
data = dict(
    dataloader="mmdet_dataloader",
    data_processor="mmdet_data_processor_dm1",
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=5,
    batch_split_nr=3,
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
work_dir="/home/wj/ai/mldata1/B11ACT/workdir/b11act_mask_yolox_huges1"
#load_from='/home/wj/ai/work/mmdetection/weights/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
load_from = '/home/wj/ai/mldata1/B11ACT/workdir/b11act_mask_yolox_huges1_fp16/weights/checkpoint_27000.pth'
#load_from = '/home/wj/ai/mldata1/B11ACT/workdir/b11act_mask_huge_fp16/weights/checkpoint1.pth'
finetune_model=True
names_not2train = ["backbone"]
names_2train = ["backbone.conv1","backbone.bn1","backbone.stem"]

