_base_ = [
    './cascade_ms.py',
    './dataset_sru2_sd4.py',
]
classes =  ('MS7U', 'MP1U', 'MU2U', 'ML9U', 'MV1U', 'ML3U', 'MS1U', 'Other')
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
        deep_stem_mode='MultiBranchStemM12X',
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        #in_channels=[96, 192, 384, 768],
        #in_channels=[128, 256, 512, 512],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type='GN',num_groups=32),
        ),
    rpn_head=dict(
        loss_obj=dict(type='WVarifocalLoss',loss_weight=5e5),
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='GenericRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=320,
            featmap_strides=[12,24],
            aggregation="concat",
            ),
        bbox_head=dict(default=dict(
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
            loss_bbox=dict(_delete_=True,type='CIoULoss', loss_weight=1.0),
            ),),
        ),
    train_cfg=dict(
        rcnn=dict(
            _delete_=True,
            default=default_rcnn_train_cfg,
            cfgs=[
                dict(
                assigner=dict(
                    pos_iou_thr=0.33,
                    neg_iou_thr=0.33,
                    min_pos_iou=0.33,
                ),),
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
            ],
         ),
        loss_scale={"s.*\.loss_cls":25,"s.*\.loss_bbox":100},
    )
)
#samples_per_gpu = 4
load_from='/home/wj/ai/work/mmdetection/weights/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_cascade_sd4i1"
names_2train = ["backbone.conv1","backbone.bn1","backbone.stem"]


