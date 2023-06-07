_base_ = [
    './cascade_ms.py',
    './dataset_sru2_sd4.py',
]
classes =  ('MS7U', 'MP1U', 'MU2U', 'ML9U', 'MV1U', 'ML3U', 'MS1U', 'Other')
model = dict(
    backbone=dict(
        _delete_=True,
        type='YOLOv8CSPDarknet',
        in_channels=1,
        #arch='tiny',
        out_indices=(1, 2, 3,4),
        deep_stem=True,
        deep_stem_mode='MultiBranchStemM12X',
        ),
    neck=dict(
        type='PAFPN',
        #in_channels=[256, 512, 1024, 2048],
        #in_channels=[96, 192, 384, 768],
        in_channels=[128, 256, 512, 512],
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
)
#samples_per_gpu = 4
load_from='yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120100-5881dec4.pth'
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_cascade_csp_sd4i"
names_2train = ["backbone.conv1","backbone.bn1","backbone.stem","backbone.norm","backbone.downsample_layers.0"]

