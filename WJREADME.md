#note
##dataset
classes index从0开始
bbox_head.num_classes不包含背景

##dataset pipline
```
resutls:
    img_info: 
        file_name: "x.jpg"
        height:
        width:
    img_prefix: img_dir,osp.join(results['img_prefix'],results['img_info']['file_name']) 应该为图像的绝对路径, 如果没有img_prefix，则直接取img_info['file_name']
    ann_info: 
         bboxes
         labels
         masks
```

### piplines

####mmdet.datasets.pipelines.loading.LoadImageFromFile

处理后增加以下字段

```
     results['filename'] = filename #文件绝对路径
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img #[H,W,C] rgb or bgr, 默认为bgr，后面normalize时会再转成rgb
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
```

####mmdet.datasets.pipelines.loading.LoadAnnotations

处理后增加以下字段

```
gt_bboxes: 从results['ann_info']['bboxes']中拷贝
gt_bboxes_ignore:[可选] 从results['ann_info']['bboxes_ignore']中拷贝
resutls['bbox_fileds']增加gt_bboxes,gt_bboxes_ignore


gt_labels: 从results['ann_info']['labels']中拷贝
gt_masks: 使用import pycocotools.mask as maskUtils生成的与原图相同大小的[NR,H,W] dtype=np.uint8, 最小值为0, 最大值为1的BitmapMasks

results['mask_fields']增加gt_masks

```
        
####mmdet.datasets.pipelines.transforms.Resize

处理后增加以下字段
```
scale: 目标大小(W,H),一般为保持比率缩放
img_shape: 更新为缩放后的大小(H,W)
pad_shape: img缩放后的大小
scale_factor: 缩放到新大小后的放大比率
keep_ratio: 是否保持缩放比率

```

####mmdet.datasets.pipelines.transforms.RandomFlip

处理后增加以下字段

```
flip: horizontal 或者 None, 表示水平翻转或不翻转
```

####mmdet.datasets.pipelines.transforms.Normalize


使用均值差进行normalize，必要时会选将bgr转换为rgb

处理后增加以下字段

```
img_norm_cfg
```

####mmdet.datasets.pipelines.transforms.Pad

将图像pad到某一个大小，如可以被32整除
将增加以下字段
```
pad_shape: 更新为新的pad后的大小
pad_fixed_size: pad到固定大小，可能为None
pad_size_divisor: pad到可被多少整除，与上面pad_fixed_size只有一个不为None
```

####mmdet.datasets.pipelines.formatting.DefaultFormatBundle

将img转换为float, 同时转换为[C,H,W], 
将img,gt_mask,gt_bboxes,gt_bboxes_ignore等再转换为tensor,再转换为DC

```
如果当前结果中没有pad_shape，就将pad_shape设置为img.shape
如果当前结果中没有scale_factor, 就将scale_factor设置为1.0
如果当前结果中没有img_norm_cfg,将将其均值设置为0，方差设置为1.0, to_rgb设置为false
```

####mmdet.datasets.pipelines.formatting.Collect


将指定的键，如('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg')对应的数据取出来
