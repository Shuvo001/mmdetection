_base_ = [
    './rcnn_ms.py'
]
model = dict(
    rpn_head=dict(
        loss_obj=dict(type='VarifocalLoss'),
    ),
)
#samples_per_gpu = 4
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_ms_ioucls"
