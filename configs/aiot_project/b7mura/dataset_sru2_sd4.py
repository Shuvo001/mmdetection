classes =  ('MS7U', 'MP1U', 'MU2U', 'ML9U', 'MV1U', 'ML3U', 'MS1U', 'Other')
label_text2id={"MP4U":7,"MU4U":None}
#MP4U
img_scale = (3840, 6144)  # height, width
random_resize_scales = [6720, 6528, 6336, 6144, 5952, 5760]
random_crop_scales = [(4006, 6720), (3892, 6528), (3777, 6336), (3663, 6144), (3548, 5952), (3434, 5760)]
img_fill_val = 255
dataset_type = 'WXMLDataset'
data_root = '/home/wj/ai/mldata1/B7mura/datas/train_sru2'
test_data_dir = '/home/wj/ai/mldata1/B7mura/datas/test_s2m'
train_pipeline = [
    dict(type='WMosaic', img_scale=img_scale, pad_val=img_fill_val,prob=0.3,skip_filter=False,two_imgs_directions=['horizontal']),
    dict(type="WRandomCrop",max_size=(4007, 6721),crop_size=random_crop_scales,name="WRandomCrop1",bbox_keep_ratio=0.001,try_crop_around_gtbboxes=True),
    dict(type='WRotate',
        prob=0.3,
        img_fill_val=img_fill_val,
        max_rotate_angle=20.0,
        ),
    dict(type='WTranslate',
        prob=0.3,
        max_translate_offset=200,
        img_fill_val=(img_fill_val,),
        ),
    dict(
        type='WMixUpWithMask',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        prob=0.1,
        pad_val=img_fill_val,skip_filter=False),
    dict(type='WResize', img_scale=random_resize_scales,multiscale_mode=True),
    #dict(type="WRandomCrop",crop_size=random_crop_scales_min,name="WRandomCrop2",bbox_keep_ratio=0.001,try_crop_around_gtbboxes=True),
    dict(type='Pad', size_divisor=192,img_fill_val=img_fill_val),
    dict(type='WFixData'),
    dict(type='DefaultFormatBundle',img_to_float=False,img_fill_val=img_fill_val),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='WLoadImageFromFile'),
    dict(type='W2Gray'),
    dict(type="WGetImg"),
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type="WResampleDataset",
        data_resample_parameters={"none":0.4},
        dataset=dict(
            type=dataset_type,
            classes=classes,
            ignored_classes=["MU4U"],
            label_text2id=label_text2id,
            img_suffix="jpg",
            ann_file=data_root,
            resample_parameters={"MS1U": 8, "ML3U": 2, "OTHER": 2, "MV1U": 2},
            pipeline=[
                dict(type='LoadImageFromFile', channel_order="rgb"),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
                dict(type='W2Gray'),
                dict(type='WResize', img_scale=img_scale),
                dict(type="WEncodeImg"),
            ],
            pipeline2=[
                dict(type="WDecodeImg", fmt='gray'),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type="WRandomChoice",transforms=[
                    (dict(type='WRandomBrightness', prob=1.0,max_delta=11),0.3),
                    (dict(type='WRandomContrast', prob=1.0,lower=0.8,upper=1.2),0.09),
                    (dict(type='WRandomBrightStripe', prob=1.0,max_size=2000),0.01),
                    ]),
            ],
            cache_file=True,
            name="b7mura_resample",
        )
    ),
    pipeline=train_pipeline,
    )
