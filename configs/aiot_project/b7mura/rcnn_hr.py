_base_ = [
    './rcnn_scale_na_pafpn.py',
]
max_iters=50000

model = dict(
    type='FasterRCNN',
    backbone=dict(
        out_indices=('stem',0,1, 2, 3),
        ),
    neck=dict(
        type='PAFPN',
        in_channels=[64,256, 512, 1024,2048],
        out_channels=128,
        num_outs=6,
    ),
    rpn_in_indices=[1,2,3,4,5],
    rpn_head=dict(
        in_channels=128,
      ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='GenericRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[12,24],
            aggregation="concat",
            ),
        bbox_head=dict(
            type='WShared4Conv2FCBBoxHead',
            in_channels=256,
            ),
        ),
    second_stage_hook=dict(type='FusionFPNHook',in_channels=128,begin_idx=1),
    drop_blocks={ "dropout":{"type":"DropBlock2D","drop_prob":[0.1,0.1,0.1,0.1,0.1,0.1],"block_size":[4,4,4,3,2,1]},
                "scheduler":{"type":"LinearScheduler","begin_step":5000,"end_step":max_iters-5000}},
)
data_root = '/home/wj/ai/mldata1/B7mura/datas/train_sr2'
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_rcnn_hr"
load_from='/home/wj/ai/work/mmdetection/weights/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'