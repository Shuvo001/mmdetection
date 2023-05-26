_base_ = [
    './rcnn_ms_sm.py',
    './dataset_sru2_sd4.py',
]
classes =  ('MS7U', 'MP1U', 'MU2U', 'ML9U', 'MV1U', 'ML3U', 'MS1U', 'Other')
model = dict(
    rpn_head=dict(
        loss_obj=dict(type='WVarifocalLoss',loss_weight=5e5),
    ),
    train_cfg=dict(
            #loss_scale={"loss_cls":25,"loss_bbox":100},
            loss_scale={"_delete_":True,"loss_cls_objectness":25,"loss_cls_classes":25,"loss_bbox":100},
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes),
            cls_predictor_cfg=dict(type='NormedLinear', tempearture=20),
            loss_cls=dict(
                type='SeesawLoss',
                p=0.8,
                q=2.0,
                num_classes=len(classes),
                loss_weight=1.0)),
    ),
)
#samples_per_gpu = 4
#load_from="/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_ms_ioucls/weights/checkpoint_15000.pth"
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_ms_sm_ren_sd4isl"
