_base_ = [
    './rcnn_ms_sm.py',
    './dataset_sru2_sd4_tiny.py',
]
classes =  ('MS7U', 'MP1U', 'MU2U', 'ML9U', 'MV1U', 'ML3U', 'MS1U', 'Other')
model = dict(
    backbone=dict(
        _delete_=True,
        type='WConvNeXt',
        in_channels=1,
        #arch='tiny',
        arch='nano',
        drop_path_rate=0.1,
        layer_scale_init_value=0.,
        out_indices=(0, 1, 2, 3),
        use_grn=True,
        deep_stem=True,
        deep_stem_mode='MultiBranchStemM12X',
        add_maxpool_after_stem=True,
        ),
    neck=dict(
        type='PAFPN',
        #in_channels=[256, 512, 1024, 2048],
        #in_channels=[96, 192, 384, 768],
        in_channels=[80, 160, 320, 640],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type='GN',num_groups=32),
        ),
    rpn_head=dict(
        loss_obj=dict(type='WVarifocalLoss',loss_weight=5e5),
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='GenericRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=336,
            featmap_strides=[12,24],
            aggregation="concat",
            ),
        bbox_head=dict(
            type='WShared4Conv2FCBBoxHead',
            norm_cfg=dict(type='GN',num_groups=32),
            in_channels=336,
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
            ),
        ),
    train_cfg=dict(
            loss_scale={"loss_cls":25,"loss_bbox":100},
    )
)
#samples_per_gpu = 4
load_from='convnext-v2-nano_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-f951ae87.pth'
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_rcnn_cn_sd4i"
names_2train = ["backbone.conv1","backbone.bn1","backbone.stem","backbone.norm","backbone.downsample_layers.0"]

