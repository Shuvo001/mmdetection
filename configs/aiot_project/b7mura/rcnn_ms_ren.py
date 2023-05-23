_base_ = [
    './rcnn_ms.py'
]
classes =  ('MS7U', 'MP1U', 'MU2U', 'ML9U', 'MV1U', 'ML3U', 'MS1U', 'Other')
img_scale = (3840, 6144)  # height, width
random_resize_scales = [6720, 6528, 6336, 6144, 5952, 5760]
random_crop_scales = [(4006, 6720), (3892, 6528), (3777, 6336), (3663, 6144), (3548, 5952), (3434, 5760)]
img_fill_val = 255
dataset_type = 'WXMLDataset'
data_root = '/home/wj/ai/mldata1/B7mura/datas/train_sru2'
test_data_dir = '/home/wj/ai/mldata1/B7mura/datas/test_s2m'
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type="WResampleDataset",
        data_resample_parameters={"none":0.4},
        dataset=dict(
            type=dataset_type,
            classes=classes,
            ignored_classes=["MU4U"],
            img_suffix="jpg",
            ann_file=data_root,
            resample_parameters={"MS1U": 8, "ML3U": 2, "Other": 2, "MV1U": 2},
            pipeline=[
                dict(type='LoadImageFromFile', channel_order="rgb"),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
                dict(type='W2Gray'),
                dict(type='WResize', img_scale=img_scale),
                dict(type="WEncodeImg"),
            ],
            pipeline2=[
                dict(type="WDecodeImg", fmt='gray'),

            ],
            cache_file=True,
            name="b7mura_resample",
        )
    ),)
#samples_per_gpu = 4
#load_from="/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_ms_ioucls/weights/checkpoint_15000.pth"
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_ms_ren"
