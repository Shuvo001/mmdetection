_base_ = [
    './rcnn_ms_sm_ren.py',
   './dataset_sru2_sd4.py',
]
#data_root = '/home/wj/ai/mldata1/B7mura/datas/try_min_bboxes_s0'
model = dict(
    rpn_head=dict(
        loss_obj=dict(type='WVarifocalLoss',loss_weight=5e5),
    ),
    train_cfg=dict(
            loss_scale={"loss_cls":25,"loss_bbox":100},
    )
)
#samples_per_gpu = 4
#load_from="/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_ms_ioucls/weights/checkpoint_15000.pth"
#load_from="/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_ms_sm_ren/weights/checkpoint_50000.pth"
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_ms_ioucls"
