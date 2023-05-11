_base_ = [
    './rcnn_ms.py'
]
model = dict(
    neck=dict(
        type='TFPN',
        num_in_transforms=2,
        num_out_transforms=1,
        ),
)
img_scale = (3840, 6144)  # height, width
random_resize_scales = [6720, 6528, 6336, 6144, 5952, 5760]
random_crop_scales = [(4006, 6720), (3892, 6528), (3777, 6336), (3663, 6144), (3548, 5952), (3434, 5760)]
img_fill_val = 255
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
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type="WRandomCrop",max_size=img_scale,crop_size=img_scale,name="WRandomCropX",bbox_keep_ratio=0.001,try_crop_around_gtbboxes=True),
    dict(type='Pad', size_divisor=192,img_fill_val=img_fill_val),
    dict(type='WFixData'),
    dict(type='DefaultFormatBundle',img_to_float=False,img_fill_val=img_fill_val),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
work_dir="/home/wj/ai/mldata1/B7mura/workdir/b7mura_faster_ms_tfpn"
