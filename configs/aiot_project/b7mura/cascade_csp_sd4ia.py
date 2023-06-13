_base_ = [
    './cascade_csp_sd4i.py',
]
#samples_per_gpu = 4
load_from='/home/wj/ai/mldata1/B7mura/workdir/b7mura_cascade_csp_sd4i/weights/latest.pth'
finetune_model=False
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_cascade_csp_sd4ia"
