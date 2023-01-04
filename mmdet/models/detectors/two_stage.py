# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
from ..builder import DETECTORS, build_backbone, build_head, build_neck,build_second_stage_hook
from .base import BaseDetector
import numpy as np
from wtorch.utils import unnormalize
import wtorch.bboxes as wtb
from mmdet.utils.datadef import *


@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 second_stage_hook=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if second_stage_hook is not None:
            self.second_stage_hook = build_second_stage_hook(second_stage_hook)
        else:
            self.second_stage_hook = None

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @staticmethod
    def recover_raw_img(img,mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True):
        '''
        return img [H,W,3] bgr order
        '''
        img = img.detach().cpu()[0]
        img = unnormalize(img,mean=mean,std=std).cpu().numpy()
        img = np.transpose(img,[1,2,0])
        if to_rgb:
            img = img[...,::-1]
        img = np.clip(img,0,255)
        img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)
        return img

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        if self.second_stage_hook is not None:
            x = self.second_stage_hook(x)
        
        _proposal_list = []
        for pbboxes,gtb in zip(proposal_list,gt_bboxes): #默认将gtbboxes加入proposal
            _proposal_list.append(self.cat_proposals_and_gtbboxes(pbboxes,gtb))

        roi_losses = self.roi_head.forward_train(x, img_metas, _proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        if is_debug():
            for k in roi_losses.keys():
                if k in losses:
                    print(f"ERROR: loss key {k} already in losses")
        losses.update(roi_losses)

        return losses

    @staticmethod
    def cat_proposals_and_gtbboxes(proposals,gtbboxes,nr=10):
        gt_nr = gtbboxes.shape[0]
        if gt_nr==0:
            return proposals
        gtbboxes = torch.unsqueeze(gtbboxes,axis=0)
        repeat_nr = int(max(1,nr/gt_nr+1))
        gtbboxes = torch.tile(gtbboxes,(repeat_nr,1,1))
        gtbboxes = torch.reshape(gtbboxes,[-1,4])
        gtbboxes = wtb.distored_boxes(gtbboxes)
        gtbboxes = gtbboxes[:nr]
        gt_nr = gtbboxes.shape[0]
        scores = gtbboxes.new_ones([gt_nr,1])

        gt_proposals = torch.cat([gtbboxes,scores],axis=-1)
        return torch.cat([gt_proposals,proposals],axis=0)

    def simple_test(self, img, img_metas, proposals=None):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            #proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            proposal_list = self.rpn_head.simple_test(x, img_metas)
        else:
            proposal_list = proposals
        
        if self.second_stage_hook is not None:
            x = self.second_stage_hook(x)

        results = self.roi_head.simple_test(
            x, proposal_list, img_metas)

        return results