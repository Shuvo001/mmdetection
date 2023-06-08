#rcnn_yoloxv2_scale.py基础上使用新的assigner, PAFPN, bnm=0.03, su1 dataset, multi scale rcnn
_base_ = [
    '../../_base_/models/cascade_rcnn_r50_fpn_yolox.py',
    '../../_base_/default_runtime.py',
    './dataset_sru2_sd4.py',
]
max_iters=50000
classes =  ('MS7U', 'MP1U', 'MU2U', 'ML9U', 'MV1U', 'ML3U', 'MS1U', 'Other')
default_bbox_head=dict(
    type='WShared4Conv2FCBBoxHead',
    norm_cfg=dict(type='GN',num_groups=32),
    in_channels=320,
    fc_out_channels=1024,
    roi_feat_size=7,
    num_shared_convs=1,
    num_classes=len(classes),
    bbox_coder=dict(
        type='DeltaXYWHBBoxCoder',
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2]),
    reg_class_agnostic=False,
    reg_decoded_bbox=True,
    loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    loss_bbox=dict(type='CIoULoss', loss_weight=1.0),
    )
default_rcnn_train_cfg=dict(
    sampler=dict(
    type='RandomSampler',
    num=128,
    pos_fraction=0.25,
    neg_pos_ub=-1,
    add_gt_as_proposals=False),
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.5,
        match_low_quality=False,
        ignore_iof_thr=-1),
    pos_weight=-1,
    debug=False,
)
# dataset settings
model = dict(
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
        deep_stem_mode='MultiBranchStemSA12X',
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type='GN',num_groups=32),
        ),
    rpn_head=dict(
        type='YOLOXRPNHead',
        in_channels=256,
        strides=[24,48,96,192,384],
        feat_channels=256,
        loss_bbox={'type': 'CIoULoss','eps': 1e-16, 'reduction': 'sum', 'loss_weight': 5.0}),
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='GenericRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=320,
            featmap_strides=[12,24],
            aggregation="concat",
            ),
        bbox_head=dict(
                _delete_=True,
                default=default_bbox_head,
                cfgs=[
                    dict(bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                ),
                dict(bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                ),
                dict(bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                ),
                ]
        ),
    ),
    second_stage_hook=dict(type='FusionFPNHook',in_channels=256,return_stem=True),
    drop_blocks={ "dropout":{"type":"DropBlock2D","drop_prob":[0.1,0.1,0.1,0.1,0.1],"block_size":[4,4,3,2,1]},
            "scheduler":{"type":"LinearScheduler","begin_step":5000,"end_step":max_iters-5000}},
    test_cfg=dict(
        rpn=dict(
            nms_pre=512,
            max_per_img=384,
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
        assigner=dict(type='SimOTAAssigner', center_radius=2.5,min_bbox_size=50),
        ),
        rcnn=dict(
            _delete_=True,
            default=default_rcnn_train_cfg,
            cfgs=[
                dict(
                assigner=dict(
                    pos_iou_thr=0.4,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                ),),
                dict(
                assigner=dict(
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                ),),
                dict(
                assigner=dict(
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                ),),
            ],
         ),
        loss_scale={"s.*\.loss_cls":25,"s.*\.loss_bbox":100},
    )
)

evaluation = dict(metric=['bbox', 'segm'])
optimizer = dict(type='SGD', momentum=0.9,nesterov=True,lr=0.001, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)
bn_momentum = 0.03
# learning policy
lr_config = dict(
    policy='WarmupCosLR',
    warmup_total_iters=1000,
    total_iters=max_iters)

log_config = dict(
    log_imgs=False,
    print_interval=10,
    tb_interval=500)
checkpoint_config = dict(
    interval=1000,
)
hooks = [
    dict(type='WMMDetModelSwitch', close_iter=-10000,skip_type_keys=('WMixUpWithMask','WRandomCrop2')),
    dict(type='WMMDetModelSwitch', close_iter=-5000,skip_type_keys=('WMosaic', 'WRandomCrop1','WRandomCrop2', 'WMixUpWithMask')),
]
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_cascade_ms"
load_from='/home/wj/ai/work/mmdetection/weights/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
#load_from = '/home/wj/ai/mldata1/B11ACT/workdir/b11act_mask_huge_fp16/weights/checkpoint.pth'
#load_from = '/home/wj/ai/mldata1/B11ACT/workdir/b11act_mask_huge_fp16/weights/checkpoint1.pth'
finetune_model=True
names_not2train = ["backbone"]
names_2train = ["backbone.conv1","backbone.bn1","backbone.stem"]

