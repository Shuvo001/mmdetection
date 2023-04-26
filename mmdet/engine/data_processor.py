from mmcv.parallel.scatter_gather import scatter_kwargs
from mmdet.datasets.pipelines.wtransforms import WCompressMask, W2PolygonMask
from object_detection2.standard_names import *
from mmdet.core import BitmapMasks
import torch
from .build import DATAPROCESSOR_REGISTRY


@DATAPROCESSOR_REGISTRY.register()
def yolo_data_processor(data_batch,local_rank=0,pipeline=None):
    inputs = {}
    inputs['img'] = data_batch[0]
    data = data_batch[1]
    nr = data_batch[2]
    gt_bboxes = []
    gt_labels = []
    img_metas = []
    batch_size = data_batch[0].shape[0]
    shape = data_batch[0].shape[2:4]
    _img_metas = {'ori_shape':shape,'img_shape':shape,'pad_shape':shape}
    for i in range(batch_size):
        _gt_bboxes = data[i,:nr[i],1:5]
        _gt_labels = data[i,:nr[i],0].to(torch.int64)
        gt_bboxes.append(_gt_bboxes)
        gt_labels.append(_gt_labels)
        img_metas.append(_img_metas)
    
        
    inputs['gt_bboxes'] = gt_bboxes
    inputs['gt_labels'] = gt_labels
    inputs['img_metas'] = img_metas

    if pipeline is not None:
        inputs = pipeline(inputs)

    return inputs

@DATAPROCESSOR_REGISTRY.register()
def mmdet_data_processor(data_batch,local_rank=0,pipeline=None):
    inputs,kwargs = scatter_kwargs(data_batch, {}, target_gpus=[local_rank], dim=0)
    inputs = inputs[0]
    if pipeline is not None:
        inputs = pipeline(inputs)
    inputs['img'] = inputs['img'].to(torch.float32)
    return inputs

@DATAPROCESSOR_REGISTRY.register()
def mmdet_data_processor_dm(data_batch,local_rank=0,pipeline=None):
    inputs,kwargs = scatter_kwargs(data_batch, {}, target_gpus=[local_rank], dim=0)
    inputs = inputs[0]
    if pipeline is not None:
        inputs = pipeline(inputs)
    inputs['img'] = inputs['img'].to(torch.float32)
    if GT_MASKS in inputs:
        new_masks = []
        for i,masks in enumerate(inputs[GT_MASKS]):
            data_nr = len(inputs[GT_BOXES][i])
            if data_nr>1:
                masks = masks.masks
                n_masks = WCompressMask.decode(masks,data_nr,True)
                n_masks = BitmapMasks(n_masks)
                new_masks.append(n_masks)
            else:
                new_masks.append(masks)
        inputs[GT_MASKS] = new_masks
    
    return inputs

@DATAPROCESSOR_REGISTRY.register()
def mmdet_data_processor_dm1(data_batch,local_rank=0,pipeline=None):
    inputs,kwargs = scatter_kwargs(data_batch, {}, target_gpus=[local_rank], dim=0)
    inputs = inputs[0]
    if pipeline is not None:
        inputs = pipeline(inputs)
    inputs['img'] = inputs['img'].to(torch.float32)
    if GT_MASKS in inputs:
        new_masks = []
        for i,masks in enumerate(inputs[GT_MASKS]):
            data_nr = len(inputs[GT_BOXES][i])
            n_masks = W2PolygonMask.decode(masks,data_nr)
            new_masks.append(n_masks)
        inputs[GT_MASKS] = new_masks
    
    return inputs
