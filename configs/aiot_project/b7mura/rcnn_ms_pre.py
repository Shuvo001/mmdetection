_base_ = [
    './rcnn_ms.py'
]
#samples_per_gpu = 4
load_from="mocov2_r50_224_epoch_200.pth"
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_ms_ylpre"
finetune_model=False