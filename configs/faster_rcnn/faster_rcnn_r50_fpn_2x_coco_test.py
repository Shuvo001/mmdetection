_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook',interval=10),
        dict(type='WTensorboardLoggerHook',interval=5,
        log_dir="/home/wj/ai/mldata/coco/tmp/coco_log",
        mean=img_norm_cfg['mean'],std=img_norm_cfg['std'],rgb=True)
    ])