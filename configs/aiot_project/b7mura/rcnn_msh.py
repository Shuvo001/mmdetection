_base_ = ['./rcnn_ms.py']
samples_per_gpu = 12
data = dict(
    samples_per_gpu=samples_per_gpu,
    )
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_msh"
load_from='/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_ms/weights/checkpoint_50000.pth'
names_2train = []
