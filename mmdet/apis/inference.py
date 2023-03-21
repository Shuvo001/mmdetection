# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from pathlib import Path
import cv2
import img_utils as wmli
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
import wtorch.utils as wtu
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
import wtorch.utils as wtu
import wml_utils as wmlu


def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmcv.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if 'pretrained' in config.model:
        config.model.pretrained = None
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None
    model = build_detector(config.model)
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint,map_location="cpu")
        wtu.forgiving_state_restore(model,checkpoint)
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """Deprecated.

    A simple pipeline to load image.
    """

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        warnings.simplefilter('once')
        warnings.warn('`LoadImage` is deprecated and will be removed in '
                      'future releases. You may use `LoadImageFromWebcam` '
                      'from `mmdet.datasets.pipelines.` instead.')
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results

def get_results(result,score_thr=0.05):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    scores = bboxes[...,-1]
    bboxes = bboxes[...,:4]
    index = scores>score_thr
    bboxes = bboxes[index]
    scores = scores[index]
    labels = labels[index]

    return bboxes,labels,scores

def inference_detectorv2(model, img,mean=None,std=None,input_size=(1024,1024),score_thr=0.05):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.
        input_size: (w,h)

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    filename = ""

    device = next(model.parameters()).device  # model device

    if not isinstance(img, np.ndarray):
        filename = img
        img = wmli.imread(img)
    
    ori_shape = [img.shape[0],img.shape[1]]
    img,r = wmli.resize_imgv2(img,input_size,return_scale=True)
    img = torch.tensor(img,dtype=torch.float32)
    img = img.permute(2,0,1)
    img = torch.unsqueeze(img,dim=0)
    img = img.to(device)

    if mean is not None:
        img = wtu.normalize(img,mean,std)
    
    img = wtu.pad_feature(img,input_size,pad_value=0)
    #img = torch.zeros_like(img) #debug

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, img=img)

    det_bboxes = results[0].cpu().numpy()
    det_masks = results[1].cpu().numpy()
    if det_masks.shape[-1]==0:
        det_masks = None
    bboxes = det_bboxes[...,:4]
    scores = det_bboxes[...,4]
    labels = det_bboxes[...,5].astype(np.int32)

    bboxes = bboxes/r

    bboxes[...,0:4:2] = np.clip(bboxes[...,0:4:2],0,ori_shape[1])
    bboxes[...,1:4:2] = np.clip(bboxes[...,1:4:2],0,ori_shape[0])
    keep0 = scores>score_thr
    keep1 = (bboxes[...,2]-bboxes[...,0])>1
    keep2 = (bboxes[...,3]-bboxes[...,1])>1
    keep = np.logical_and(keep0,keep1)
    keep = np.logical_and(keep,keep2)
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if det_masks is not None:
        det_masks = det_masks[keep]

    return bboxes,labels,scores,det_masks,results

def inference_traced_detector(model, img,mean=None,std=None,input_size=(1024,1024),score_thr=0.05,device=torch.device("cuda")):
    """Inference image(s) with the detector.

    Args:
        model traced model
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.
        input_size: (w,h)

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    is_batch = False
    filename = ""

    if not isinstance(img, np.ndarray):
        filename = img
        img = wmli.imread(img)
    
    ori_shape = [img.shape[0],img.shape[1]]
    img,r = wmli.resize_imgv2(img,input_size,return_scale=True)
    img = torch.tensor(img,dtype=torch.float32)
    img = img.permute(2,0,1)
    img = torch.unsqueeze(img,dim=0)
    img = img.to(device)

    if mean is not None:
        img = wtu.normalize(img,mean,std)
    
    img = wtu.pad_feature(img,input_size,pad_value=0)

    with torch.no_grad():
        results = model(img)

    det_bboxes = results[0].cpu().numpy()
    det_masks = results[1].cpu().numpy()
    if det_masks.shape[-1]==0:
        det_masks = None
    bboxes = det_bboxes[...,:4]
    scores = det_bboxes[...,4]
    labels = det_bboxes[...,5].astype(np.int32)

    bboxes = bboxes/r

    bboxes[...,0:4:2] = np.clip(bboxes[...,0:4:2],0,ori_shape[1])
    bboxes[...,1:4:2] = np.clip(bboxes[...,1:4:2],0,ori_shape[0])
    keep0 = scores>score_thr
    keep1 = (bboxes[...,2]-bboxes[...,0])>1
    keep2 = (bboxes[...,3]-bboxes[...,1])>1
    keep = np.logical_and(keep0,keep1)
    keep = np.logical_and(keep,keep2)
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if det_masks is not None:
        det_masks = det_masks[keep]

    return bboxes,labels,scores,det_masks,results

async def async_inference_detector(model, imgs):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.

    Returns:
        Awaitable detection results.
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    results = await model.aforward_test(rescale=True, **data)
    return results


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       title='result',
                       wait_time=0,
                       palette=None,
                       out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param. Default: 0.
        palette (str or tuple(int) or :obj:`Color`): Color.
            The tuple of color should be in BGR order.
        out_file (str or None): The path to write the image.
            Default: None.
    """
    if hasattr(model, 'module'):
        model = model.module
    return model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=palette,
        text_color=(200, 200, 200),
        mask_color=palette,
        out_file=out_file)

class ImageInferencePipeline:
    def __init__(self,pipeline) -> None:
        self.pipeline = pipeline
        self.time = wmlu.AvgTimeThis()
        pass

    def __call__(self,model, img,input_size=(1024,1024),score_thr=0.05):
        """Inference image(s) with the detector.

        Args:
            model (nn.Module): The loaded detector.
            imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
               Either image files or loaded images.
            input_size: (w,h)
    
        Returns:
            If imgs is a list or tuple, the same length list type results
            will be returned, otherwise return the detection results directly.
        """
    
        filename = ""
    
        device = next(model.parameters()).device  # model device
    
        if not isinstance(img, np.ndarray):
            filename = img
            img = wmli.imread(img)
        
        ori_shape = [img.shape[0],img.shape[1]]
        img,r = wmli.resize_imgv2(img,input_size,return_scale=True)
        img = self.pipeline(img)
        img = torch.tensor(img,dtype=torch.float32)
        img = img.permute(2,0,1)
        img = torch.unsqueeze(img,dim=0)
        img = img.to(device)
    
        
        img = wtu.pad_feature(img,input_size,pad_value=0)
        #img = torch.zeros_like(img) #debug
    
        # forward the model
        with torch.no_grad():
            with self.time:
                results = model(return_loss=False, img=img)
    
        det_bboxes = results[0].cpu().numpy()
        det_masks = results[1].cpu().numpy()
        if det_masks.shape[0]==0:
            det_masks = None
        bboxes = det_bboxes[...,:4]
        scores = det_bboxes[...,4]
        labels = det_bboxes[...,5].astype(np.int32)
    
        bboxes = bboxes/r
    
        bboxes[...,0:4:2] = np.clip(bboxes[...,0:4:2],0,ori_shape[1])
        bboxes[...,1:4:2] = np.clip(bboxes[...,1:4:2],0,ori_shape[0])
        keep0 = scores>score_thr
        keep1 = (bboxes[...,2]-bboxes[...,0])>1
        keep2 = (bboxes[...,3]-bboxes[...,1])>1
        keep = np.logical_and(keep0,keep1)
        keep = np.logical_and(keep,keep2)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if det_masks is not None:
            det_masks = det_masks[keep]
    
        return bboxes,labels,scores,det_masks,results
    
    def __del__(self):
        print(self.time)