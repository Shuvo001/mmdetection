_base_ = [
    './rcnn_ms_sm.py',
    './dataset_sru2_sd4_tiny.py',
]
model = dict(
    backbone=dict(
        _delete_=True,
        type='WCSPResNet',
        input_channels=1,
        deepen_factor= 1.0,
        widen_factor= 1.0,
        #out_indices=(0, 1, 2, 3),
        out_indices=(1, 2, 3,4),
        deep_stem=True,
        deep_stem_mode='MultiBranchStemM12X',
        ),
    neck=dict(
        type='PAFPN',
        #in_channels=[256, 512, 1024, 2048],
        in_channels=[128,256, 512, 1024],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type='GN',num_groups=32),
        ),
    rpn_head=dict(
        loss_obj=dict(type='WVarifocalLoss',loss_weight=5e5),
    ),
    train_cfg=dict(
            loss_scale={"loss_cls":25,"loss_bbox":100},
            max_norm=0.0,
    )
)
#samples_per_gpu = 4
finetune_model=False
#load_from='ppyoloe_plus_l_fast_8xb8-80e_coco_20230102_203825-1864e7b3.pth'
load_from='/home/wj/ai/mldata1/B7mura/workdir/b7mura_rcnn_cr_sd4i/weights/checkpoint_5000.pth'
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_rcnn_cr_sd4i"
