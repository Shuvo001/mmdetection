import copy
import cv2
import mmcv
import numpy as np
from ..builder import PIPELINES
import random
from mmdet.core import find_inside_bboxes, BitmapMasks,PolygonMasks
import wtorch.utils as wtu
import img_utils as wmli
import object_detection2.bboxes as odb
from object_detection2.standard_names import *
from collections import Iterable
import torch
import math

@PIPELINES.register_module()
class WChannelSpatialRandomCrop:
    '''
    通过在通道或空间维度随机裁减来减小显存消耗
    '''

    def __init__(self,
                 mem_reduce=0.25,
                 channel_mem_reduce_prob=0.5,
                 bbox_keep_ratio=0.25,
                 try_crop_around_gtbboxes=True,
                 crop_around_gtbboxes_prob=0.5,
                 name='WChannelSpatialRandomCrop'):
        self.mem_reduce = mem_reduce
        self.channel_mem_reduce_prob = channel_mem_reduce_prob
        self.bbox_keep_ratio = bbox_keep_ratio
        self.try_crop_around_gtbboxes = try_crop_around_gtbboxes
        self.crop_around_gtbboxes_prob = crop_around_gtbboxes_prob
        self.name = name

    def get_crop_bbox(self,crop_size,img_shape,gtbboxes):
        '''
        crop_size: (h,w)
        img_shape: (H,W)
        '''
        max_len_gtbboxes = max([len(x) for x in gtbboxes])
        if not self.try_crop_around_gtbboxes or gtbboxes is None or max_len_gtbboxes==0:
            return self.get_random_crop_bbox(crop_size,img_shape)
        if np.random.rand() > self.crop_around_gtbboxes_prob:
            return self.get_random_crop_bbox(crop_size,img_shape)

        return self.random_crop_around_gtbboxes(crop_size,img_shape,gtbboxes)

    def get_random_crop_bbox(self,crop_size,img_shape):
        h, w = img_shape
        new_h = min(crop_size[0],h)
        new_w = min(crop_size[1],w)
        if new_w<w:
            x_offset = np.random.randint(low=0, high=w - new_w)
        else:
            x_offset = 0
        if new_h<h:
            y_offset = np.random.randint(low=0, high=h - new_h)
        else:
            y_offset = 0

        patch = np.array([x_offset, y_offset,x_offset+new_w,y_offset+new_h],dtype=np.int32)

        return patch

    def random_crop_around_gtbboxes(self,crop_size,img_shape,gtbboxes):
        bboxes = torch.cat(gtbboxes,axis=0)
        h, w = img_shape
        try:
            bbox = random.choice(bboxes)
            cx = random.randint(bbox[0],bbox[2]-1)
            cy = random.randint(bbox[1],bbox[3]-1)
        except:
            cx = bbox[0]
            cy = bbox[1]
        x_offset = max(cx-crop_size[1]//2,0)
        y_offset = max(cy-crop_size[0]//2,0)

        x_offset = max(min(x_offset,w-crop_size[1]),0)
        y_offset = max(min(y_offset,h-crop_size[0]),0)

        new_h = min(crop_size[0],h-y_offset)
        new_w = min(crop_size[1],w-x_offset)

        patch = np.array([x_offset, y_offset,x_offset+new_w,y_offset+new_h],dtype=np.int32)

        return patch



    def _train_aug(self, results):
        """Random crop and around padding the original image.

        Args:
            results (dict): Image infomations in the augment pipeline.

        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        B,C,H,W = img.shape
        spatial_scale = math.sqrt(self.mem_reduce)
        nH = int(H*spatial_scale)
        nW = int(W*spatial_scale)
        crop_size = (nH,nW)
        gtbboxes = results.get('gt_bboxes',None)
        patch = self.get_crop_bbox(crop_size,(H,W),gtbboxes)
        try:
            cropped_img = wmli.crop_batch_img_absolute_xy(img, patch)
        except:
            print("Crop error:",patch)

        results['img'] = cropped_img
        results['img_shape'] = cropped_img.shape
        results['pad_shape'] = cropped_img.shape
        x_offset = patch[0]
        y_offset = patch[1]
        new_w = cropped_img.shape[1]
        new_h = cropped_img.shape[0]

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', ['gt_bboxes']):
            bboxes = results[key]
            bboxes,keep = self.crop_batch_bboxes(bboxes,x_offset,y_offset,new_w,new_h)
            results[key] = bboxes
            if key in ['gt_bboxes']:
                if 'gt_labels' in results:
                    labels = results['gt_labels']
                    labels = [ls[kp] for ls,kp in zip(labels,keep)]
                    results['gt_labels'] = labels
                if 'gt_masks' in results:
                    gt_masks = results['gt_masks'].masks
                    gt_masks = [gm[kp] for gm,kp in zip(gt_masks,keep)]
                    gt_masks = [wmli.crop_masks_absolute_xy(gm,patch) for gm in gt_masks]
                    results['gt_masks'] = [BitmapMasks(gm) for gm in gt_masks]

            # crop semantic seg
            for key in results.get('seg_fields', []):
                raise NotImplementedError(
                    'RandomCenterCropPad only supports bbox.')
            if IMG_METAS in results:
                for x in results[IMG_METAS]:
                    x['img_shape'] = list(results[IMAGE].shape[-2:])+[results[IMAGE].shape[1]]
            return results

    def crop_batch_bboxes(self,bboxes,x_offset,y_offset,new_w,new_h):
        res_bboxes = []
        res_keep = []
        for bbox in bboxes:
            bbox,keep = self.crop_bboxes(bbox,x_offset,y_offset,new_w,new_h)
            res_bboxes.append(bbox)
            res_keep.append(keep)
        return res_bboxes,res_keep


    def crop_bboxes(self,bboxes,x_offset,y_offset,new_w,new_h):
        old_bboxes = copy.deepcopy(bboxes)
        old_area = odb.torch_area(old_bboxes)
        bboxes[:, 0:4:2] -= x_offset
        bboxes[:, 1:4:2] -= y_offset
        bboxes[:, 0:4:2] = torch.clip(bboxes[:, 0:4:2], 0, new_w)
        bboxes[:, 1:4:2] = torch.clip(bboxes[:, 1:4:2], 0, new_h)
        keep0 = (bboxes[:, 2] > bboxes[:, 0]) & (
            bboxes[:, 3] > bboxes[:, 1])
        new_area = odb.torch_area(bboxes)
        area_ratio = new_area/(old_area+1e-6)
        keep1 = area_ratio>self.bbox_keep_ratio
        keep = torch.logical_and(keep0,keep1)
        bboxes = bboxes[keep]
        return bboxes,keep

    def _channel_train_aug(self, results):
        """Random crop and around padding the original image.

        Args:
            results (dict): Image infomations in the augment pipeline.

        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        batch_size = img.shape[0]
        new_batch_size = int(batch_size*self.mem_reduce)
        batch_offset = int(np.random.rand()*(batch_size-new_batch_size))
        img = img[batch_offset:batch_offset+new_batch_size]
        if IMG_METAS in results:
            results[IMG_METAS] = results[IMG_METAS][batch_offset:batch_offset+new_batch_size]

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', ['gt_bboxes']):
            bboxes = results[key]
            bboxes = bboxes[batch_offset:batch_offset+new_batch_size]
            results[key] = bboxes
            if key in ['gt_bboxes']:
                if 'gt_labels' in results:
                    labels = results['gt_labels']
                    labels = labels[batch_offset:batch_offset+new_batch_size]
                    results['gt_labels'] = labels
                if 'gt_masks' in results:
                    gt_masks = results['gt_masks'].masks
                    gt_masks = gt_masks[batch_offset:batch_offset+new_batch_size]
                    results['gt_masks'] = BitmapMasks(gt_masks)

            # crop semantic seg
            for key in results.get('seg_fields', []):
                raise NotImplementedError(
                    'RandomCenterCropPad only supports bbox.')
            return results

    def __call__(self, results):
        if np.random.rand()<self.channel_mem_reduce_prob:
            res = self._channel_train_aug(results)
        else:
            res = self._train_aug(results)
        torch.cuda.empty_cache()
        return res

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(bbox keep ratio ={self.bbox_keep_ratio}, '
        return repr_str

@PIPELINES.register_module()
class WExpandBatchByRandomCrop:
    '''
    通过在通道或空间维度随机裁减来减小显存消耗
    '''

    def __init__(self,
                 spatial_mem_reduce=0.25,
                 expand_batch_prob=0.5,
                 bbox_keep_ratio=0.25,
                 try_crop_around_gtbboxes=True,
                 crop_around_gtbboxes_prob=0.5,
                 name='WExpandBatchByRandomCrop'):
        self.mem_reduce = spatial_mem_reduce
        self.expand_batch_prob = expand_batch_prob 
        self.bbox_keep_ratio = bbox_keep_ratio
        self.try_crop_around_gtbboxes = try_crop_around_gtbboxes
        self.crop_around_gtbboxes_prob = crop_around_gtbboxes_prob
        self.name = name

    def get_crop_bbox(self,crop_size,img_shape,gtbboxes):
        '''
        crop_size: (h,w)
        img_shape: (H,W)
        '''
        max_len_gtbboxes = max([len(x) for x in gtbboxes])
        if not self.try_crop_around_gtbboxes or gtbboxes is None or max_len_gtbboxes==0:
            return self.get_random_crop_bbox(crop_size,img_shape)
        if np.random.rand() > self.crop_around_gtbboxes_prob:
            return self.get_random_crop_bbox(crop_size,img_shape)

        return self.random_crop_around_gtbboxes(crop_size,img_shape,gtbboxes)

    def get_random_crop_bbox(self,crop_size,img_shape):
        h, w = img_shape
        new_h = min(crop_size[0],h)
        new_w = min(crop_size[1],w)
        if new_w<w:
            x_offset = np.random.randint(low=0, high=w - new_w)
        else:
            x_offset = 0
        if new_h<h:
            y_offset = np.random.randint(low=0, high=h - new_h)
        else:
            y_offset = 0

        patch = np.array([x_offset, y_offset,x_offset+new_w,y_offset+new_h],dtype=np.int32)

        return patch

    def random_crop_around_gtbboxes(self,crop_size,img_shape,gtbboxes):
        bboxes = torch.cat(gtbboxes,axis=0)
        h, w = img_shape
        try:
            bbox = random.choice(bboxes)
            cx = random.randint(bbox[0],bbox[2]-1)
            cy = random.randint(bbox[1],bbox[3]-1)
        except:
            cx = bbox[0]
            cy = bbox[1]
        x_offset = max(cx-crop_size[1]//2,0)
        y_offset = max(cy-crop_size[0]//2,0)

        x_offset = max(min(x_offset,w-crop_size[1]),0)
        y_offset = max(min(y_offset,h-crop_size[0]),0)

        new_h = min(crop_size[0],h-y_offset)
        new_w = min(crop_size[1],w-x_offset)

        patch = np.array([x_offset, y_offset,x_offset+new_w,y_offset+new_h],dtype=np.int32)

        return patch

    def train_aug(self, results):
        nr = int(1.0/self.mem_reduce+0.1)
        a_results = []
        new_results = {}
        for i in range(nr):
            a_results.append(self._train_aug(copy.deepcopy(results)))
        cat_keys = [IMAGE]
        for k in results:
            if k in cat_keys:
                new_results[k] = torch.cat([x[k] for x in a_results],dim=0)
            else:
                nd = []
                for x in a_results:
                    nd.extend(x[k])
                new_results[k] = nd
        return new_results

    def _train_aug(self, results):
        """Random crop and around padding the original image.

        Args:
            results (dict): Image infomations in the augment pipeline.

        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        B,C,H,W = img.shape
        spatial_scale = math.sqrt(self.mem_reduce)
        nH = int(H*spatial_scale)
        nW = int(W*spatial_scale)
        crop_size = (nH,nW)
        gtbboxes = results.get('gt_bboxes',None)
        patch = self.get_crop_bbox(crop_size,(H,W),gtbboxes)
        try:
            cropped_img = wmli.crop_batch_img_absolute_xy(img, patch)
        except:
            print("Crop error:",patch)

        results['img'] = cropped_img
        results['img_shape'] = cropped_img.shape
        results['pad_shape'] = cropped_img.shape
        x_offset = patch[0]
        y_offset = patch[1]
        new_w = cropped_img.shape[1]
        new_h = cropped_img.shape[0]

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', ['gt_bboxes']):
            bboxes = results[key]
            bboxes,keep = self.crop_batch_bboxes(bboxes,x_offset,y_offset,new_w,new_h)
            results[key] = bboxes
            if key in ['gt_bboxes']:
                if 'gt_labels' in results:
                    labels = results['gt_labels']
                    labels = [ls[kp] for ls,kp in zip(labels,keep)]
                    results['gt_labels'] = labels
                if 'gt_masks' in results:
                    gt_masks = results['gt_masks'].masks
                    gt_masks = [gm[kp] for gm,kp in zip(gt_masks,keep)]
                    gt_masks = [wmli.crop_masks_absolute_xy(gm,patch) for gm in gt_masks]
                    results['gt_masks'] = [BitmapMasks(gm) for gm in gt_masks]

            # crop semantic seg
            for key in results.get('seg_fields', []):
                raise NotImplementedError(
                    'RandomCenterCropPad only supports bbox.')
            if IMG_METAS in results:
                for x in results[IMG_METAS]:
                    x['img_shape'] = list(results[IMAGE].shape[-2:])+[results[IMAGE].shape[1]]
            return results

    def crop_batch_bboxes(self,bboxes,x_offset,y_offset,new_w,new_h):
        res_bboxes = []
        res_keep = []
        for bbox in bboxes:
            bbox,keep = self.crop_bboxes(bbox,x_offset,y_offset,new_w,new_h)
            res_bboxes.append(bbox)
            res_keep.append(keep)
        return res_bboxes,res_keep


    def crop_bboxes(self,bboxes,x_offset,y_offset,new_w,new_h):
        old_bboxes = copy.deepcopy(bboxes)
        old_area = odb.torch_area(old_bboxes)
        bboxes[:, 0:4:2] -= x_offset
        bboxes[:, 1:4:2] -= y_offset
        bboxes[:, 0:4:2] = torch.clip(bboxes[:, 0:4:2], 0, new_w)
        bboxes[:, 1:4:2] = torch.clip(bboxes[:, 1:4:2], 0, new_h)
        keep0 = (bboxes[:, 2] > bboxes[:, 0]) & (
            bboxes[:, 3] > bboxes[:, 1])
        new_area = odb.torch_area(bboxes)
        area_ratio = new_area/(old_area+1e-6)
        keep1 = area_ratio>self.bbox_keep_ratio
        keep = torch.logical_and(keep0,keep1)
        bboxes = bboxes[keep]
        return bboxes,keep

    def __call__(self, results):
        if np.random.rand()<self.expand_batch_prob:
            results = self.train_aug(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(bbox keep ratio ={self.bbox_keep_ratio}, '
        return repr_str