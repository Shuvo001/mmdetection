_base_ = [
    './rcnn_ms.py'
]
model = dict(
    rpn_head=dict(
        loss_obj=dict(type='VarifocalLoss'),
    ),
    train_cfg=dict(
            loss_scale={"loss_cls":25,"loss_bbox":100,"loss_yolox_obj":1e5},
    )
)
#samples_per_gpu = 4
#load_from="/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_ms_ioucls/weights/checkpoint_15000.pth"
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_ms_ioucls"
