# Copyright (c) OpenMMLab. All rights reserved.
import sys
from inspect import signature

import torch
from mmcv.ops import batched_nms

from mmdet.core import bbox_mapping_back, merge_aug_proposals

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


class BBoxTestMixin(object):
    """Mixin class for testing det bboxes via DenseHead."""

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        outs = self.forward(feats)
        results_list = self.get_bboxes(
            *outs, img_metas=img_metas, rescale=rescale)
        return results_list

    def simple_test_rpn(self, x, img_metas):
        """Test without augmentation, only for ``RPNHead`` and its variants,
        e.g., ``GARPNHead``, etc.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        rpn_outs = self(x) #调用另一个基类, BaseDenseHead.forward, 如rpn_head调用AnchorHead.forward
        proposal_list = self.get_bboxes(*rpn_outs, img_metas=img_metas) #调用另一个基类,BaseDenseHead.get_bboxes
        return proposal_list
