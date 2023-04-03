from img_patch import ImagePatch
import object_detection2.bboxes as odb
import warnings
from pathlib import Path
import cv2
import img_utils as wmli
import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
import wtorch.utils as wtu
from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.pipelines.wtransforms import WGradAguImg
from mmdet.models import build_detector
import wtorch.utils as wtu
import os.path as osp
import torchvision
import sys

def make_new_img(img):
    img = img[...,::-1]
    img = WGradAguImg.apply(img)
    img = img[...,::-1]
    img = np.ascontiguousarray(img)
    return img

class ImagePatchInference:
    def __init__(self,patch_size,pad=True,pad_value=227,boundary=128,img_process=None,use_gray_img=False) -> None:
        self.image_patch = ImagePatch(patch_size=patch_size,
                                      pad=pad,
                                      pad_value=pad_value,boundary=boundary)
        self.img_process = img_process
        self.use_gray_img = use_gray_img

    def __call__(self,model, img,mean=None,std=None,input_size=(1024,1024),score_thr=0.05,iou_threshold=0.3):
        if not isinstance(img, np.ndarray):
            img = wmli.imread(img)
        self.image_patch.set_src_img(img)
        all_bboxes = []
        all_labels = []
        all_scores = []
        all_det_masks = []

        #save_dir = "/home/wj/ai/mldata1/B11ACT/workdir/b11act_mask_mosaic_patch/tmp/eval_on_images/patchs"
        #wmli.imwrite(osp.join(save_dir,"raw.jpg"),img)
        for i in range(len(self.image_patch)):
            sys.stdout.write(f"{i}/{len(self.image_patch)}    \r")
            patch_img = self.image_patch[i]
            if self.img_process is not None:
                patch_img = self.img_process(patch_img)
            bboxes,labels,scores,det_masks,results =self.inference_detector(model,
                                                                            patch_img,
                                                                            mean,
                                                                            std,
                                                                            input_size,
                                                                            score_thr=score_thr
                                                                            )
            #
            #wmli.imwrite(osp.join(save_dir,f"{i}.jpg"),patch_img)
            #
            keep = self.image_patch.remove_boundary_bboxes(bboxes)
            bboxes = bboxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            bboxes = self.image_patch.patch_bboxes2img_bboxes(bboxes)
            all_bboxes.append(bboxes)
            all_labels.append(labels)
            all_scores.append(scores)
            if det_masks is not None:
                det_masks = det_masks[keep]
                all_det_masks.append(det_masks)
        #print("") 
        #print(save_dir)
        if len(all_bboxes)==0:
            return np.zeros([0,4],dtype=np.float32),np.zeros([0],dtype=np.int32),np.zeros([0],dtype=np.float32),None,None
        else:
            all_bboxes = np.concatenate(all_bboxes,axis=0)
            all_labels = np.concatenate(all_labels,axis=0)
            all_scores = np.concatenate(all_scores,axis=0)
            if len(all_det_masks)>0:
                all_det_masks = np.concatenate(all_det_masks,axis=0)
            else:
                all_det_masks = None

            idx = self.nms(all_bboxes,all_scores,threshold=iou_threshold) 
            all_bboxes = all_bboxes[idx]
            all_labels = all_labels[idx]
            all_scores = all_scores[idx]
            if all_det_masks is not None:
                all_det_masks = all_det_masks[idx]
            idx = self.bboxes_area_nms(all_bboxes,all_scores,threshold=iou_threshold) 
            all_bboxes = all_bboxes[idx]
            all_labels = all_labels[idx]
            all_scores = all_scores[idx]
            if all_det_masks is not None:
                all_det_masks = all_det_masks[idx]
            return all_bboxes,all_labels,all_scores,all_det_masks,None

    def inference_detector(self,model, img,mean=None,std=None,input_size=(1024,1024),score_thr=0.05):
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
    
        device = next(model.parameters()).device  # model device
    
        ori_shape = [img.shape[0],img.shape[1]]
        if input_size is not None:
            img,r = wmli.resize_imgv2(img,input_size,return_scale=True)
        else:
            r = 1.0
        img = torch.tensor(img,dtype=torch.float32)
        img = img.permute(2,0,1)
        img = torch.unsqueeze(img,dim=0)
        img = img.to(device)
    
        if mean is not None:
            img = wtu.normalize(img,mean,std)
            
        if self.use_gray_img:
            img = wtu.rgb2gray(img)
        
        if input_size is not None:
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

    def nms(self,bboxes,scores,threshold=0.5):
        bboxes = torch.from_numpy(bboxes)
        scores = torch.from_numpy(scores)
        idx = torchvision.ops.nms(bboxes,scores,iou_threshold=threshold)
        idx = idx.cpu().numpy()
        return idx

    def bboxes_area_nms(self,bboxes,scores,threshold=0.5):
        if len(bboxes) == 0:
            return np.zeros([0],dtype=np.int32)
        bboxes_area = odb.area(bboxes)
        idxs = np.argsort(bboxes_area)[::-1]
        keep = np.ones([bboxes.shape[0]],dtype=np.bool)

        for i,idx in enumerate(idxs[:-1]):
            cur_bbox = bboxes[idx]
            cur_idxs = idxs[i+1:]
            cur_keep = keep[cur_idxs]
            cur_idxs = cur_idxs[cur_keep]
            cur_bboxes = bboxes[cur_idxs]

            ious = odb.npbboxes_intersection_of_box0(cur_bboxes,np.expand_dims(cur_bbox,0))
            t_v = ious>threshold
            t_idx = cur_idxs[t_v]
            keep[t_idx] = False
        
        return np.nonzero(keep)

class ImagePatchInferencePipeline:
    def __init__(self,patch_size,
                      pad=True,pad_value=227,boundary=128,
                      pipeline=None) -> None:
        self.image_patch = ImagePatch(patch_size=patch_size,
                                      pad=pad,
                                      pad_value=pad_value,boundary=boundary)
        self.pipeline = pipeline

    def __call__(self,model, img,input_size=(1024,1024),score_thr=0.05,iou_threshold=0.3):
        if not isinstance(img, np.ndarray):
            img = wmli.imread(img)
        self.image_patch.set_src_img(img)
        all_bboxes = []
        all_labels = []
        all_scores = []
        all_det_masks = []

        #save_dir = "/home/wj/ai/mldata1/B11ACT/workdir/b11act_mask_mosaic_patch/tmp/eval_on_images/patchs"
        #wmli.imwrite(osp.join(save_dir,"raw.jpg"),img)
        for i in range(len(self.image_patch)):
            sys.stdout.write(f"{i}/{len(self.image_patch)}    \r")
            patch_img = self.image_patch[i]
            bboxes,labels,scores,det_masks,results =self.inference_detector(model,
                                                                            patch_img,
                                                                            input_size,
                                                                            score_thr=score_thr
                                                                            )
            #
            #wmli.imwrite(osp.join(save_dir,f"{i}.jpg"),patch_img)
            #
            keep = self.image_patch.remove_boundary_bboxes(bboxes)
            bboxes = bboxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            bboxes = self.image_patch.patch_bboxes2img_bboxes(bboxes)
            all_bboxes.append(bboxes)
            all_labels.append(labels)
            all_scores.append(scores)
            if det_masks is not None:
                det_masks = det_masks[keep]
                all_det_masks.append(det_masks)
        #print("") 
        #print(save_dir)
        if len(all_bboxes)==0:
            return np.zeros([0,4],dtype=np.float32),np.zeros([0],dtype=np.int32),np.zeros([0],dtype=np.float32),None,None
        else:
            all_bboxes = np.concatenate(all_bboxes,axis=0)
            all_labels = np.concatenate(all_labels,axis=0)
            all_scores = np.concatenate(all_scores,axis=0)
            if len(all_det_masks)>0:
                all_det_masks = np.concatenate(all_det_masks,axis=0)
            else:
                all_det_masks = None

            idx = self.nms(all_bboxes,all_scores,threshold=iou_threshold) 
            all_bboxes = all_bboxes[idx]
            all_labels = all_labels[idx]
            all_scores = all_scores[idx]
            if all_det_masks is not None:
                all_det_masks = all_det_masks[idx]
            idx = self.bboxes_area_nms(all_bboxes,all_scores,threshold=iou_threshold) 
            all_bboxes = all_bboxes[idx]
            all_labels = all_labels[idx]
            all_scores = all_scores[idx]
            if all_det_masks is not None:
                all_det_masks = all_det_masks[idx]
            return all_bboxes,all_labels,all_scores,all_det_masks,None

    def inference_detector(self,model, img,input_size=(1024,1024),score_thr=0.05):
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
    
        device = next(model.parameters()).device  # model device
        ori_shape = [img.shape[0],img.shape[1]]
        img = self.pipeline(img)
        if input_size is not None:
            img,r = wmli.resize_imgv2(img,input_size,return_scale=True)
        else:
            r = 1.0
        img = torch.tensor(img,dtype=torch.float32)
        img = img.permute(2,0,1)
        img = torch.unsqueeze(img,dim=0)
        img = img.to(device)
    
        
        if input_size is not None:
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

    def nms(self,bboxes,scores,threshold=0.5):
        bboxes = torch.from_numpy(bboxes)
        scores = torch.from_numpy(scores)
        idx = torchvision.ops.nms(bboxes,scores,iou_threshold=threshold)
        idx = idx.cpu().numpy()
        return idx

    def bboxes_area_nms(self,bboxes,scores,threshold=0.5):
        if len(bboxes) == 0:
            return np.zeros([0],dtype=np.int32)
        bboxes_area = odb.area(bboxes)
        idxs = np.argsort(bboxes_area)[::-1]
        keep = np.ones([bboxes.shape[0]],dtype=np.bool)

        for i,idx in enumerate(idxs[:-1]):
            cur_bbox = bboxes[idx]
            cur_idxs = idxs[i+1:]
            cur_keep = keep[cur_idxs]
            cur_idxs = cur_idxs[cur_keep]
            cur_bboxes = bboxes[cur_idxs]

            ious = odb.npbboxes_intersection_of_box0(cur_bboxes,np.expand_dims(cur_bbox,0))
            t_v = ious>threshold
            t_idx = cur_idxs[t_v]
            keep[t_idx] = False
        
        return np.nonzero(keep)


class ImagePatchInferencePipeline:
    def __init__(self,patch_size,
                      pad=True,img_fill_val=0,boundary=0,
                      pipeline=None) -> None:
        self.image_patch = ImagePatch(patch_size=patch_size,
                                      pad=pad,
                                      pad_value=img_fill_val,boundary=boundary)
        self.pipeline = pipeline

    def __call__(self,model, img,input_size=None,score_thr=0.05,iou_threshold=0.3):
        if not isinstance(img, np.ndarray):
            img = wmli.imread(img)
        self.image_patch.set_src_img(img)

        #save_dir = "/home/wj/ai/mldata1/B11ACT/workdir/b11act_mask_mosaic_patch/tmp/eval_on_images/patchs"
        #wmli.imwrite(osp.join(save_dir,"raw.jpg"),img)
        for i in range(len(self.image_patch)):
            sys.stdout.write(f"{i}/{len(self.image_patch)}    \r")
            patch_img = self.image_patch[i]
            bboxes,labels,scores,det_masks,results =self.inference_detector(model,
                                                                            patch_img,
                                                                            input_size,
                                                                            score_thr=score_thr
                                                                            )
            yield bboxes,labels,scores,det_masks,self.image_patch.cur_bbox()

    def inference_detector(self,model, img,input_size=(1024,1024),score_thr=0.05):
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
    
        device = next(model.parameters()).device  # model device
        ori_shape = [img.shape[0],img.shape[1]]
        img = self.pipeline(img)
        if input_size is not None:
            img,r = wmli.resize_imgv2(img,input_size,return_scale=True)
        else:
            r = 1.0
        img = torch.tensor(img,dtype=torch.float32)
        img = img.permute(2,0,1)
        img = torch.unsqueeze(img,dim=0)
        img = img.to(device)
    
        
        if input_size is not None:
            img = wtu.pad_feature(img,input_size,pad_value=0)
        #img = torch.zeros_like(img) #debug
    
        # forward the model
        with torch.no_grad():
            results = model(return_loss=False, img=img)
    
        det_bboxes = results[0].cpu().numpy()
        det_masks = results[1].cpu().numpy()
        if det_masks.size == 0:
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

    def nms(self,bboxes,scores,threshold=0.5):
        bboxes = torch.from_numpy(bboxes)
        scores = torch.from_numpy(scores)
        idx = torchvision.ops.nms(bboxes,scores,iou_threshold=threshold)
        idx = idx.cpu().numpy()
        return idx

    def bboxes_area_nms(self,bboxes,scores,threshold=0.5):
        if len(bboxes) == 0:
            return np.zeros([0],dtype=np.int32)
        bboxes_area = odb.area(bboxes)
        idxs = np.argsort(bboxes_area)[::-1]
        keep = np.ones([bboxes.shape[0]],dtype=np.bool)

        for i,idx in enumerate(idxs[:-1]):
            cur_bbox = bboxes[idx]
            cur_idxs = idxs[i+1:]
            cur_keep = keep[cur_idxs]
            cur_idxs = cur_idxs[cur_keep]
            cur_bboxes = bboxes[cur_idxs]

            ious = odb.npbboxes_intersection_of_box0(cur_bboxes,np.expand_dims(cur_bbox,0))
            t_v = ious>threshold
            t_idx = cur_idxs[t_v]
            keep[t_idx] = False
        
        return np.nonzero(keep)

class ImagePatchInferencePipelineV2:
    def __init__(self,patch_size,
                      pad=True,pad_value=227,boundary=128,
                      pipeline=None) -> None:
        self.image_patch = ImagePatch(patch_size=patch_size,
                                      pad=pad,
                                      pad_value=pad_value,boundary=boundary)
        self.pipeline = pipeline

    def __call__(self,model, img,input_size=(1024,1024),score_thr=0.05,iou_threshold=0.3):
        if not isinstance(img, np.ndarray):
            img = wmli.imread(img)
        self.image_patch.set_src_img(img)
        all_bboxes = []
        all_labels = []
        all_scores = []
        all_det_masks = []

        #save_dir = "/home/wj/ai/mldata1/B11ACT/workdir/b11act_mask_mosaic_patch/tmp/eval_on_images/patchs"
        #wmli.imwrite(osp.join(save_dir,"raw.jpg"),img)
        for i in range(len(self.image_patch)):
            sys.stdout.write(f"{i}/{len(self.image_patch)}    \r")
            patch_img = self.image_patch[i]
            bboxes,labels,scores,det_masks,results =self.inference_detector(model,
                                                                            patch_img,
                                                                            input_size,
                                                                            score_thr=score_thr
                                                                            )
            #
            #wmli.imwrite(osp.join(save_dir,f"{i}.jpg"),patch_img)
            #
            keep = self.image_patch.remove_boundary_bboxes(bboxes)
            bboxes = bboxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            bboxes = self.image_patch.patch_bboxes2img_bboxes(bboxes)
            all_bboxes.append(bboxes)
            all_labels.append(labels)
            all_scores.append(scores)
            if det_masks is not None:
                det_masks = det_masks[keep]
                all_det_masks.append(det_masks)
        #print("") 
        #print(save_dir)
        if len(all_bboxes)==0:
            return np.zeros([0,4],dtype=np.float32),np.zeros([0],dtype=np.int32),np.zeros([0],dtype=np.float32),None,None
        else:
            all_bboxes = np.concatenate(all_bboxes,axis=0)
            all_labels = np.concatenate(all_labels,axis=0)
            all_scores = np.concatenate(all_scores,axis=0)
            if len(all_det_masks)>0:
                all_det_masks = np.concatenate(all_det_masks,axis=0)
            else:
                all_det_masks = None

            idx = self.nms(all_bboxes,all_scores,threshold=iou_threshold) 
            all_bboxes = all_bboxes[idx]
            all_labels = all_labels[idx]
            all_scores = all_scores[idx]
            if all_det_masks is not None:
                all_det_masks = all_det_masks[idx]
            idx = self.bboxes_area_nms(all_bboxes,all_scores,threshold=iou_threshold) 
            all_bboxes = all_bboxes[idx]
            all_labels = all_labels[idx]
            all_scores = all_scores[idx]
            if all_det_masks is not None:
                all_det_masks = all_det_masks[idx]
            return all_bboxes,all_labels,all_scores,all_det_masks,None

    def inference_detector(self,model, img,input_size=(1024,1024),score_thr=0.05):
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
    
        device = next(model.parameters()).device  # model device
        ori_shape = [img.shape[0],img.shape[1]]
        img = self.pipeline(img)
        if input_size is not None:
            img,r = wmli.resize_imgv2(img,input_size,return_scale=True)
        else:
            r = 1.0
        img = torch.tensor(img,dtype=torch.float32)
        img = img.permute(2,0,1)
        img = torch.unsqueeze(img,dim=0)
        img = img.to(device)
    
        
        if input_size is not None:
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

    def nms(self,bboxes,scores,threshold=0.5):
        bboxes = torch.from_numpy(bboxes)
        scores = torch.from_numpy(scores)
        idx = torchvision.ops.nms(bboxes,scores,iou_threshold=threshold)
        idx = idx.cpu().numpy()
        return idx

    def bboxes_area_nms(self,bboxes,scores,threshold=0.5):
        if len(bboxes) == 0:
            return np.zeros([0],dtype=np.int32)
        bboxes_area = odb.area(bboxes)
        idxs = np.argsort(bboxes_area)[::-1]
        keep = np.ones([bboxes.shape[0]],dtype=np.bool)

        for i,idx in enumerate(idxs[:-1]):
            cur_bbox = bboxes[idx]
            cur_idxs = idxs[i+1:]
            cur_keep = keep[cur_idxs]
            cur_idxs = cur_idxs[cur_keep]
            cur_bboxes = bboxes[cur_idxs]

            ious = odb.npbboxes_intersection_of_box0(cur_bboxes,np.expand_dims(cur_bbox,0))
            t_v = ious>threshold
            t_idx = cur_idxs[t_v]
            keep[t_idx] = False
        
        return np.nonzero(keep)
