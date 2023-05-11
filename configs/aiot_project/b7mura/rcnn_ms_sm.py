##rcnn_ms.py基础上使用新的stem

_base_ = [
    './rcnn_ms.py'
]
model = dict(
    backbone=dict(
        deep_stem_mode='MultiBranchStemM12X',
        ),
)
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_ms_sm"
load_from='/home/wj/ai/work/mmdetection/weights/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'