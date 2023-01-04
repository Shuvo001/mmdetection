# Copyright (c) OpenMMLab. All rights reserved.
import torch
import warnings
from mmdet.core import bbox2result, bbox2roi,bbox2roi_one_img, build_assigner, build_sampler
from mmdet.core.bbox.transforms import bbox2result_yolo_style
import math
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
import numpy as np
from mmdet.utils.datadef import *


@HEADS.register_module()
class StandardRoIHead(BaseRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            #bbox_assigner默认为MaxIoUAssigner in mmdet/core/bbox/assigners/max_iou_assigner.py
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            #bbox_sampler 默认为 mmdet.core.bbox.samplers.random_sampler.RandomSampler
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        #bbox_roi_extractor default is mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor.SingleRoIExtractor
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        #bbox_head default is mmdet.models.roi_heads.bbox_heads.convfc_bbox_head.Shared2FCBBoxHead
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            #bbox_roi_extractor default is mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor.SingleRoIExtractor
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        rcn_pos_inds_nr = 0
        rcn_neg_inds_nr = 0
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
                rcn_pos_inds_nr += sampling_result.pos_inds.numel()
                rcn_neg_inds_nr += sampling_result.neg_inds.numel()

        losses = dict()

        #for log
        losses['rcn_pos_inds_nr'] = torch.tensor(rcn_pos_inds_nr,dtype=torch.float32).cuda()
        losses['rcn_neg_inds_nr'] = torch.tensor(rcn_neg_inds_nr,dtype=torch.float32).cuda()
        #end for log

        if is_debug():
            if self.with_bbox:
                nr = min(len(x),len(self.bbox_roi_extractor.roi_layers))
                fw_h,fw_w = img_metas[0]['forward_shape'][-2:]
                for i in range(nr):
                    delta = 1.0/self.bbox_roi_extractor.roi_layers[i].spatial_scale
                    fm_h,fm_w = x[i].shape[-2:]
                    if math.fabs(fm_h*delta-fw_h)>delta or \
                        math.fabs(fm_w*delta-fw_w)>delta:
                        print(f"ERROR bbox roialign spatial scale value {self.bbox_roi_extractor.roi_layers[i].spatial_scale} \
                            for layer {i}, expected {fm_h/fw_h,fm_w/fw_w}")
            if self.with_mask and not self.share_roi_extractor:
                nr = min(len(x),len(self.mask_roi_extractor.roi_layers))
                fw_h,fw_w = img_metas[0]['forward_shape'][-2:]
                for i in range(nr):
                    delta = 1.0/self.mask_roi_extractor.roi_layers[i].spatial_scale
                    fm_h,fm_w = x[i].shape[-2:]
                    if math.fabs(fm_h*delta-fw_h)>delta or \
                        math.fabs(fm_w*delta-fw_w)>delta:
                        print(f"ERROR mask roialign spatial scale value {self.mask_roi_extractor.roi_layers[i].spatial_scale} for layer {i},\
                             expected {fm_h/fw_h,fm_w/fw_w}")



        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois) #如果feature maps(即x)的数量与self.bbox_roi_extractor设置的数量不一致,取前self.bbox_roi_extractor.num_inputs个
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
    
    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info. 仅使用image_shape, 用于约束输出bboxes的范围
            proposals (List[Tensor]): tensor [N,5], (x0,y0,x1,y1,score) Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        '''if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size'''

        bbox_results = self._bbox_forward(x, rois)
        if img_metas is not None:
            #wj debug
            img_shapes = (img_metas[0].get('img_shape',None),)*len(img_metas)
        else:
            img_shapes = None

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        num_proposals_per_img = [] 
        for p in proposals:
            num_proposals_per_img.append(len(p)) 

        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0) #split to each img

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor): #default True
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)): #for each image
            det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i], #指定输出bbox的最大值
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None):
        """Test without augmentation.
        ***This function one support test one image.***

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w). tuple corresponding to different feature level
                from hight resolution to low resolution
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score). list corresponding to dirrerent images
            img_metas (list[dict]): Meta information of images.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg)

        i = 0
        bbox_results = bbox2result_yolo_style(det_bboxes[i], det_labels[i])

        if not self.with_mask:
            return bbox_results,torch.zeros([0,28,28],dtype=torch.uint8)
        else:
            segm_results = self.simple_test_mask(
                x, det_bboxes, det_labels)
            return bbox_results,segm_results
    
    def simple_test__(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w). tuple corresponding to different feature level
                from hight resolution to low resolution
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score). list corresponding to dirrerent images
            img_metas (list[dict]): Meta information of images.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg)

        max_bboxes_nr = 100
        bbox_results = det_bboxes[0].new_zeros([len(det_bboxes),max_bboxes_nr,6])
        for i in range(len(det_bboxes)):
            result = bbox2result_yolo_style(det_bboxes[i], det_labels[i])
            nr = min(max_bboxes_nr,result.shape[0])
            bbox_results[i,:nr] = result[:nr]

        if not self.with_mask:
            return bbox_results,torch.zeros([0,0,0],dtype=torch.uint8)
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels)
            return bbox_results,segm_results

    def simple_test_mask(self,
                         x,
                         det_bboxes,
                         det_labels):
        """Simple test for mask head without augmentation."""
        '''This function only support test one img'''

        _bboxes = det_bboxes[0][...,:4]
        mask_rois = bbox2roi_one_img(_bboxes,0)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        segm_results = self.mask_head.get_seg_masks(
                mask_pred, _bboxes, det_labels[0],
                self.test_cfg)
        return segm_results


    def simple_test_mask__(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels):
        """Simple test for mask head without augmentation."""
        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [torch.zeros([0,0,0],dtype=torch.uint8) for x in det_bboxes]
        else:
            _bboxes = [det_bbox[...,:4] for det_bbox in det_bboxes]
            mask_rois = bbox2roi(_bboxes)
            mask_results = self._mask_forward(x, mask_rois)
            mask_pred = mask_results['mask_pred']
            # split batch mask prediction back to each image
            num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
            mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

            # apply mask post-processing to each image individually
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append(torch.zeros([0,0,0],dtype=torch.uint8))
                        
                else:
                    segm_result = self.mask_head.get_seg_masks(
                        mask_preds[i], _bboxes[i], det_labels[i],
                        self.test_cfg)
                    segm_results.append(segm_result)
        return segm_results