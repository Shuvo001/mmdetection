_base_ = [
    './rcnn_ms.py',
]
model = dict(
    type='FasterRCNN',
    backbone=dict(
        _delete_=True,
        type='WCSPResNet',
        input_channels=1,
        deepen_factor= 1.0,
        widen_factor= 1.0,
        #out_indices=(0, 1, 2, 3),
        out_indices=(1, 2, 3, 4),
        deep_stem=True,
        deep_stem_mode='MultiBranchStemSA12X',
        ),
    neck=dict(
        in_channels=[128,256, 512, 1024],
        ),
)
load_from='ppyoloe_plus_l_fast_8xb8-80e_coco_20230102_203825-1864e7b3.pth'
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_ms_cr"