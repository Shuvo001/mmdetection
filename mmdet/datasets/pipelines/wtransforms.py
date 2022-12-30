import copy
import cv2
import mmcv
import numpy as np
import torch
from ..builder import PIPELINES
from .compose import Compose
import random
from mmdet.core import find_inside_bboxes, BitmapMasks,PolygonMasks
import wtorch.utils as wtu
import img_utils as wmli
import object_detection2.bboxes as odb
from object_detection2.standard_names import *
from collections import Iterable



def bbox2fields():
    """The key correspondence from bboxes to labels, masks and
    segmentations."""
    bbox2label = {
        'gt_bboxes': 'gt_labels',
        'gt_bboxes_ignore': 'gt_labels_ignore'
    }
    bbox2mask = {
        'gt_bboxes': 'gt_masks',
        'gt_bboxes_ignore': 'gt_masks_ignore'
    }
    bbox2seg = {
        'gt_bboxes': 'gt_semantic_seg',
    }
    return bbox2label, bbox2mask, bbox2seg

@PIPELINES.register_module()
class WRandomCrop:
    '''
    '''

    def __init__(self,
                 crop_size=None, #(H,W) or list of (H,W)
                 img_pad_value=127,
                 mask_pad_value=0,
                 bbox_keep_ratio=0.25,
                 crop_if=None,
                 try_crop_around_gtbboxes=False,
                 crop_around_gtbboxes_prob=0.5,
                 name='WRandomCrop'):
        assert isinstance(crop_size, (list, tuple))
        self.crop_size = crop_size
        self.img_pad_value = img_pad_value
        self.mask_pad_value = mask_pad_value
        self.bbox_keep_ratio = bbox_keep_ratio
        self.crop_if = crop_if
        self.try_crop_around_gtbboxes = try_crop_around_gtbboxes
        self.crop_around_gtbboxes_prob = crop_around_gtbboxes_prob
        self.multiscale_mode = isinstance(crop_size[0],Iterable)
        self.name = name

    def get_crop_bbox(self,crop_size,img_shape,gtbboxes):
        '''
        crop_size: (h,w)
        '''
        if not self.try_crop_around_gtbboxes or gtbboxes is None or len(gtbboxes)==0:
            return self.get_random_crop_bbox(crop_size,img_shape)
        if np.random.rand() > self.crop_around_gtbboxes_prob:
            return self.get_random_crop_bbox(crop_size,img_shape)

        return self.random_crop_around_gtbboxes(crop_size,img_shape,gtbboxes)

    def get_random_crop_bbox(self,crop_size,img_shape):
        h, w, c = img_shape
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

        patch = [x_offset, y_offset,x_offset+new_w,y_offset+new_h]

        return patch

    def random_crop_around_gtbboxes(self,crop_size,img_shape,gtbboxes):
        h, w, c = img_shape

        bbox = random.choice(gtbboxes)
        cx = random.randint(bbox[0],bbox[2]-1)
        cy = random.randint(bbox[1],bbox[3]-1)
        x_offset = max(cx-crop_size[1]//2,0)
        y_offset = max(cy-crop_size[0]//2,0)

        x_offset = max(min(x_offset,w-crop_size[1]),0)
        y_offset = max(min(y_offset,h-crop_size[0]),0)

        new_h = min(crop_size[0],h)
        new_w = min(crop_size[1],w)

        patch = [x_offset, y_offset,x_offset+new_w,y_offset+new_h]

        return patch



    def _train_aug(self, results):
        """Random crop and around padding the original image.

        Args:
            results (dict): Image infomations in the augment pipeline.

        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        if not self.multiscale_mode:
            crop_size = self.crop_size
        else:
            crop_size = random.choice(self.crop_size)
        gtbboxes = results.get('gt_bboxes',None)
        patch = self.get_crop_bbox(crop_size,img.shape,gtbboxes)
        cropped_img = wmli.crop_img_absolute_xy(img, patch)

        x_offset = patch[0]
        y_offset = patch[1]
        new_w = patch[2]-x_offset
        new_h = patch[3]-y_offset
        results['img'] = cropped_img
        results['img_shape'] = cropped_img.shape
        results['pad_shape'] = cropped_img.shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            old_bboxes = copy.deepcopy(bboxes)
            old_area = odb.area(old_bboxes)
            bboxes[:, 0:4:2] -= x_offset
            bboxes[:, 1:4:2] -= y_offset
            bboxes[:, 0:4:2] = np.clip(bboxes[:, 0:4:2], 0, new_w)
            bboxes[:, 1:4:2] = np.clip(bboxes[:, 1:4:2], 0, new_h)
            keep0 = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            new_area = odb.area(bboxes)
            area_ratio = new_area/(old_area+1e-6)
            keep1 = area_ratio>self.bbox_keep_ratio
            keep = np.logical_and(keep0,keep1)
            bboxes = bboxes[keep]
            results[key] = bboxes
            if key in ['gt_bboxes']:
                if 'gt_labels' in results:
                    labels = results['gt_labels']
                    labels = labels[keep]
                    results['gt_labels'] = labels
                if 'gt_masks' in results:
                    gt_masks = results['gt_masks'].masks
                    gt_masks = gt_masks[keep]
                    gt_masks = wmli.crop_masks_absolute_xy(gt_masks,patch)
                    results['gt_masks'] = BitmapMasks(gt_masks)

            # crop semantic seg
            for key in results.get('seg_fields', []):
                raise NotImplementedError(
                    'RandomCenterCropPad only supports bbox.')
            return results

    def __call__(self, results):
        if self.crop_if is not None:
            process_pipline = results.get('process',[])
            for key in self.crop_if:
                if key in process_pipline:
                    return self._train_aug(results)
            return results
        else:
            return self._train_aug(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'(bbox keep ratio ={self.bbox_keep_ratio}, '
        return repr_str


@PIPELINES.register_module()
class WRotate:
    """Apply Rotate Transformation to image (and its corresponding bbox, mask,
    segmentation).

    Args:
        level (int | float): The level should be in range (0,_MAX_LEVEL].
        scale (int | float): Isotropic scale factor. Same in
            ``mmcv.imrotate``.
        center (int | float | tuple[float]): Center point (w, h) of the
            rotation in the source image. If None, the center of the
            image will be used. Same in ``mmcv.imrotate``.
        img_fill_val (int | float | tuple): The fill value for image border.
            If float, the same value will be used for all the three
            channels of image. If tuple, the should be 3 elements (e.g.
            equals the number of channels for image).
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        prob (float): The probability for perform transformation and
            should be in range 0 to 1.
        max_rotate_angle (int | float): The maximum angles for rotate
            transformation.
        random_negative_prob (float): The probability that turns the
             offset negative.
    """

    def __init__(self,
                 scale=1,
                 center=None,
                 img_fill_val=128,
                 seg_ignore_label=255,
                 prob=0.5,
                 max_rotate_angle=30):
        assert isinstance(scale, (int, float)), \
            f'The scale must be type int or float. got type {type(scale)}.'
        if isinstance(center, (int, float)):
            center = (center, center)
        elif isinstance(center, tuple):
            assert len(center) == 2, 'center with type tuple must have '\
                f'2 elements. got {len(center)} elements.'
        else:
            assert center is None, 'center must be None or type int, '\
                f'float or tuple, got type {type(center)}.'
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, 'img_fill_val as tuple must '\
                f'have 3 elements. got {len(img_fill_val)}.'
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError(
                'img_fill_val must be float or tuple with 3 elements.')
        assert np.all([0 <= val <= 255 for val in img_fill_val]), \
            'all elements of img_fill_val should between range [0,255]. '\
            f'got {img_fill_val}.'
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. '\
            f'got {prob}.'
        assert isinstance(max_rotate_angle, (int, float)), 'max_rotate_angle '\
            f'should be type int or float. got type {type(max_rotate_angle)}.'
        self.scale = scale
        # Rotation angle in degrees. Positive values mean
        # clockwise rotation.
        self.angle =  max_rotate_angle
        self.center = center
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob
        self.max_rotate_angle = max_rotate_angle

    def _rotate_img(self, results, angle, center=None, scale=1.0):
        """Rotate the image.

        Args:
            results (dict): Result dict from loading pipeline.
            angle (float): Rotation angle in degrees, positive values
                mean clockwise rotation. Same in ``mmcv.imrotate``.
            center (tuple[float], optional): Center point (w, h) of the
                rotation. Same in ``mmcv.imrotate``.
            scale (int | float): Isotropic scale factor. Same in
                ``mmcv.imrotate``.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            img_rotated = mmcv.imrotate(
                img, angle, center, scale, border_value=self.img_fill_val)
            if len(img_rotated.shape)==2:
                img_rotated = np.expand_dims(img_rotated,axis=-1)
            results[key] = img_rotated.astype(img.dtype)
            results['img_shape'] = results[key].shape

    def _rotate_bboxes(self, results, rotate_matrix):
        """Rotate the bboxes."""
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1)
            coordinates = np.stack([[min_x, min_y], [max_x, min_y],
                                    [min_x, max_y],
                                    [max_x, max_y]])  # [4, 2, nb_bbox, 1]
            # pad 1 to convert from format [x, y] to homogeneous
            # coordinates format [x, y, 1]
            coordinates = np.concatenate(
                (coordinates,
                 np.ones((4, 1, coordinates.shape[2], 1), coordinates.dtype)),
                axis=1)  # [4, 3, nb_bbox, 1]
            coordinates = coordinates.transpose(
                (2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
            rotated_coords = np.matmul(rotate_matrix,
                                       coordinates)  # [nb_bbox, 4, 2, 1]
            rotated_coords = rotated_coords[..., 0]  # [nb_bbox, 4, 2]
            min_x, min_y = np.min(
                rotated_coords[:, :, 0], axis=1), np.min(
                    rotated_coords[:, :, 1], axis=1)
            max_x, max_y = np.max(
                rotated_coords[:, :, 0], axis=1), np.max(
                    rotated_coords[:, :, 1], axis=1)
            min_x, min_y = np.clip(
                min_x, a_min=0, a_max=w), np.clip(
                    min_y, a_min=0, a_max=h)
            max_x, max_y = np.clip(
                max_x, a_min=min_x, a_max=w), np.clip(
                    max_y, a_min=min_y, a_max=h)
            results[key] = np.stack([min_x, min_y, max_x, max_y],
                                    axis=-1).astype(results[key].dtype)

    def _rotate_masks(self,
                      results,
                      angle,
                      center=None,
                      scale=1.0,
                      fill_val=0):
        """Rotate the masks."""
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.rotate((h, w), angle, center, scale, fill_val)

    def _rotate_seg(self,
                    results,
                    angle,
                    center=None,
                    scale=1.0,
                    fill_val=255):
        """Rotate the segmentation map."""
        for key in results.get('seg_fields', []):
            seg = results[key].copy()
            results[key] = mmcv.imrotate(
                seg, angle, center, scale,
                border_value=fill_val).astype(seg.dtype)

    def _filter_invalid(self, results, min_bbox_size=0):
        """Filter bboxes and corresponding masks too small after rotate
        augmentation."""
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_inds = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_inds = np.nonzero(valid_inds)[0]
            results[key] = results[key][valid_inds]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds]

    def __call__(self, results):
        """Call function to rotate images, bounding boxes, masks and semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """
        if np.random.rand() > self.prob:
            return results
        h, w = results['img'].shape[:2]
        center = self.center
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        angle = np.random.uniform(-self.angle, self.angle)
        self._rotate_img(results, angle, center, self.scale)
        rotate_matrix = cv2.getRotationMatrix2D(center, -angle, self.scale)
        self._rotate_bboxes(results, rotate_matrix)
        self._rotate_masks(results, angle, center, self.scale, fill_val=0)
        self._rotate_seg(
            results, angle, center, self.scale, fill_val=self.seg_ignore_label)
        self._filter_invalid(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'scale={self.scale}, '
        repr_str += f'center={self.center}, '
        repr_str += f'img_fill_val={self.img_fill_val}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'max_rotate_angle={self.max_rotate_angle}, '
        return repr_str


@PIPELINES.register_module()
class WTranslate:
    """Translate the images, bboxes, masks and segmentation maps horizontally
    or vertically.

    Args:
        level (int | float): The level for Translate and should be in
            range [0,_MAX_LEVEL].
        prob (float): The probability for performing translation and
            should be in range [0, 1].
        img_fill_val (int | float | tuple): The filled value for image
            border. If float, the same fill value will be used for all
            the three channels of image. If tuple, the should be 3
            elements (e.g. equals the number of channels for image).
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        direction (str): The translate direction, either "horizontal"
            or "vertical".
        max_translate_offset (int | float): The maximum pixel's offset for
            Translate.
        min_size (int | float): The minimum pixel for filtering
            invalid bboxes after the translation.
    """

    def __init__(self,
                 prob=0.5,
                 img_fill_val=128,
                 seg_ignore_label=255,
                 directions='horizontal',
                 max_translate_offset=250.,
                 min_size=0):
        assert 0 <= prob <= 1.0, \
            'The probability of translation should be in range [0, 1].'
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError('img_fill_val must be type float or tuple.')
        assert np.all([0 <= val <= 255 for val in img_fill_val]), \
            'all elements of img_fill_val should between range [0,255].'
        #assert direction in ('horizontal', 'vertical'), \
            #'direction should be "horizontal" or "vertical".'
        assert isinstance(max_translate_offset, (int, float)), \
            'The max_translate_offset must be type int or float.'
        # the offset used for translation
        self.offset = max_translate_offset
        self.prob = prob
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.directions = directions
        self._direction = None
        self.max_translate_offset = max_translate_offset
        self.min_size = min_size

    def _translate_img(self, results, offset, direction='horizontal'):
        """Translate the image.

        Args:
            results (dict): Result dict from loading pipeline.
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
        """
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            img = mmcv.imtranslate(
                img, offset, direction, self.img_fill_val).astype(img.dtype)
            if len(img.shape) == 2:
                img = np.expand_dims(img,axis=-1)
            results[key] = img
            results['img_shape'] = results[key].shape

    def _translate_bboxes(self, results, offset):
        """Shift bboxes horizontally or vertically, according to offset."""
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1)
            if self._direction == 'horizontal':
                min_x = np.maximum(0, min_x + offset)
                max_x = np.minimum(w, max_x + offset)
            elif self._direction == 'vertical':
                min_y = np.maximum(0, min_y + offset)
                max_y = np.minimum(h, max_y + offset)

            # the boxes translated outside of image will be filtered along with
            # the corresponding masks, by invoking ``_filter_invalid``.
            results[key] = np.concatenate([min_x, min_y, max_x, max_y],
                                          axis=-1)

    def _translate_masks(self,
                         results,
                         offset,
                         direction='horizontal',
                         fill_val=0):
        """Translate masks horizontally or vertically."""
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.translate((h, w), offset, direction, fill_val)

    def _translate_seg(self,
                       results,
                       offset,
                       direction='horizontal',
                       fill_val=255):
        """Translate segmentation maps horizontally or vertically."""
        for key in results.get('seg_fields', []):
            seg = results[key].copy()
            results[key] = mmcv.imtranslate(seg, offset, direction,
                                            fill_val).astype(seg.dtype)

    def _filter_invalid(self, results, min_size=0):
        """Filter bboxes and masks too small or translated out of image."""
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_inds = (bbox_w > min_size) & (bbox_h > min_size)
            valid_inds = np.nonzero(valid_inds)[0]
            results[key] = results[key][valid_inds]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds]
        return results

    def __call__(self, results):
        """Call function to translate images, bounding boxes, masks and
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Translated results.
        """
        if np.random.rand() > self.prob:
            return results
        offset = np.random.uniform(-self.offset,self.offset)
        if isinstance(self.directions,str):
            self._direction = self.directions
        else:
            self._direction = random.choice(self.directions)
        self._translate_img(results, offset, self._direction)
        self._translate_bboxes(results, offset)
        # fill_val defaultly 0 for BitmapMasks and None for PolygonMasks.
        self._translate_masks(results, offset, self._direction)
        # fill_val set to ``seg_ignore_label`` for the ignored value
        # of segmentation map.
        self._translate_seg(
            results, offset, self._direction, fill_val=self.seg_ignore_label)
        self._filter_invalid(results, min_size=self.min_size)
        return results

@PIPELINES.register_module()
class WMixUpWithMask:
    """MixUp data augmentation.

    .. code:: text

                         mixup transform
                +------------------------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                |---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      |-----------------+     |
                |             pad              |
                +------------------------------+

     The mixup transform steps are as follows:

        1. Another random image is picked by dataset and embedded in
           the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
           image and origin image.

    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (height, width). Default: (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Default: (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Default: 0.5.
        pad_val (int): Pad value. Default: 114.
        max_iters (int): The maximum number of iterations. If the number of
            iterations is greater than `max_iters`, but gt_bbox is still
            empty, then the iteration is terminated. Default: 15.
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 5.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed. Default: 20.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` and `min_area_ratio` and `max_aspect_ratio`
            is invalid. Default to True.
    """

    def __init__(self,
                 img_scale=(640, 640), #(H,W)
                 ratio_range=(0.5, 1.5),
                 flip_ratio=0.5,
                 pad_val=114,
                 max_iters=15,
                 min_bbox_size=5,
                 min_area_ratio=0.2,
                 max_aspect_ratio=20,
                 bbox_clip_border=True,
                 skip_filter=True,
                 prob=1.0):
        assert isinstance(img_scale, tuple)
        self.dynamic_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter
        self.prob = prob

    def __call__(self, results):
        """Call function to make a mixup of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mixup transformed.
        """
        process_pipline = results.get('process',[])

        if 'WMosaic' in process_pipline:
            prob = self.prob/2
        else:
            prob = self.prob

        if random.uniform(0, 1) > prob:
            return results

        results = self._mixup_transform(results)
        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        return random.randint(0, len(dataset)-1)
        '''for i in range(self.max_iters):
            index = random.randint(0, len(dataset)-1)
            gt_bboxes_i = dataset.get_ann_info(index)['bboxes']
            if len(gt_bboxes_i) != 0:
                break

        return index'''

    def _mixup_transform(self, results):
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        assert len(
            results['mix_results']) == 1, 'MixUp only support 2 images now !'

        if results['mix_results'][0]['gt_bboxes'].shape[0] == 0:
            # empty bbox
            return results

        retrieve_results = results['mix_results'][0]
        retrieve_img = retrieve_results['img']
        img_channels = retrieve_img.shape[-1]
        if 'gt_masks' in retrieve_results:
            retrieve_masks = retrieve_results['gt_masks'].masks
            retrieve_bboxes = retrieve_results[GT_BOXES]
            out_masks = []
        else:
            retrieve_masks = None

        jit_factor = random.uniform(*self.ratio_range)
        is_filp = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones(
                (self.dynamic_scale[0], self.dynamic_scale[1], retrieve_img.shape[-1]),
                dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val
        
        if retrieve_masks is not None:
            retrieve_masks_shape = (self.dynamic_scale[0], self.dynamic_scale[1])

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0],
                          self.dynamic_scale[1] / retrieve_img.shape[1])
        retrieve_img = wmli.resize_img(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                           int(retrieve_img.shape[0] * scale_ratio)))
        

        # 2. paste
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

        if retrieve_masks is not None:
            #retrieve_masks = wtu.npresize_mask(retrieve_masks, (retrieve_img.shape[1], retrieve_img.shape[0]))
            retrieve_masks,retrieve_bboxes = wtu.npresize_mask_in_bboxes(retrieve_masks, retrieve_bboxes,(retrieve_img.shape[1], retrieve_img.shape[0]))
            n_mask_i = np.zeros([retrieve_masks.shape[0],retrieve_masks_shape[0],retrieve_masks_shape[1]],dtype=retrieve_masks.dtype)
            n_mask_i[:,:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_masks 

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = wmli.resize_img(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))
        if retrieve_masks is not None:
            #n_mask_i = wtu.npresize_mask(n_mask_i, (out_img.shape[1], out_img.shape[0]))
            n_mask_i,_ = wtu.npresize_mask_in_bboxes(n_mask_i,retrieve_bboxes, (out_img.shape[1], out_img.shape[0]))

        # 4. flip
        if is_filp:
            out_img = out_img[:, ::-1, :]
            if retrieve_masks is not None:
                n_mask_i = n_mask_i[:,:,::-1]

        # 5. random crop
        ori_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w,
                                          target_w), img_channels)).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img
        if retrieve_masks is not None:
            nn_mask_i = np.zeros((n_mask_i.shape[0],max(origin_h, target_h), max(origin_w,
                                          target_w)),dtype=n_mask_i.dtype)
            nn_mask_i[:,:origin_h, :origin_w] = n_mask_i
            

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]
        if retrieve_masks is not None:
            nn_mask_i = nn_mask_i[:,y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes[:, 0::2] = retrieve_gt_bboxes[:, 0::2] * scale_ratio
        retrieve_gt_bboxes[:, 1::2] = retrieve_gt_bboxes[:, 1::2] * scale_ratio
        if self.bbox_clip_border:
            retrieve_gt_bboxes[:, 0::2] = np.clip(retrieve_gt_bboxes[:, 0::2],
                                                  0, origin_w)
            retrieve_gt_bboxes[:, 1::2] = np.clip(retrieve_gt_bboxes[:, 1::2],
                                                  0, origin_h)

        if is_filp:
            retrieve_gt_bboxes[:, 0::2] = (
                origin_w - retrieve_gt_bboxes[:, 0::2][:, ::-1])

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
        cp_retrieve_gt_bboxes[:, 0::2] = \
            cp_retrieve_gt_bboxes[:, 0::2] - x_offset
        cp_retrieve_gt_bboxes[:, 1::2] = \
            cp_retrieve_gt_bboxes[:, 1::2] - y_offset
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes[:, 0::2] = np.clip(
                cp_retrieve_gt_bboxes[:, 0::2], 0, target_w)
            cp_retrieve_gt_bboxes[:, 1::2] = np.clip(
                cp_retrieve_gt_bboxes[:, 1::2], 0, target_h)

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_labels = retrieve_results['gt_labels']
        if not self.skip_filter:
            keep_list = self._filter_box_candidates(retrieve_gt_bboxes.T,
                                                    cp_retrieve_gt_bboxes.T)

            retrieve_gt_labels = retrieve_gt_labels[keep_list]
            cp_retrieve_gt_bboxes = cp_retrieve_gt_bboxes[keep_list]
            if retrieve_masks is not None:
                nn_mask_i = nn_mask_i[keep_list]

        mixup_gt_bboxes = np.concatenate(
            (results['gt_bboxes'], cp_retrieve_gt_bboxes), axis=0)
        mixup_gt_labels = np.concatenate(
            (results['gt_labels'], retrieve_gt_labels), axis=0)
        if retrieve_masks is not None:
            mixup_gt_masks = np.concatenate([results['gt_masks'].masks,nn_mask_i],axis=0)

        # remove outside bbox
        inside_inds = find_inside_bboxes(mixup_gt_bboxes, target_h, target_w)
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_labels = mixup_gt_labels[inside_inds]
        if retrieve_masks is not None:
            mixup_gt_masks = mixup_gt_masks[inside_inds]
            results['gt_masks'] = BitmapMasks(mixup_gt_masks)

        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_labels'] = mixup_gt_labels

        return results

    def _filter_box_candidates(self, bbox1, bbox2):
        """Compute candidate boxes which include following 5 things:

        bbox1 before augment, bbox2 after augment, min_bbox_size (pixels),
        min_area_ratio, max_aspect_ratio.
        """

        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return ((w2 > self.min_bbox_size)
                & (h2 > self.min_bbox_size)
                & (w2 * h2 / (w1 * h1 + 1e-16) > self.min_area_ratio)
                & (ar < self.max_aspect_ratio))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'dynamic_scale={self.dynamic_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'max_iters={self.max_iters}, '
        repr_str += f'min_bbox_size={self.min_bbox_size}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_aspect_ratio={self.max_aspect_ratio}, '
        repr_str += f'skip_filter={self.skip_filter})'
        return repr_str

@PIPELINES.register_module()
class WResize:
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None, #(H,W)
                 multiscale_mode=False,
                 bbox_clip_border=True,
                 override=False):

        if multiscale_mode:
            if not isinstance(img_scale[0],Iterable):
                self.img_scale = [(x,x) for x in img_scale]
            else:
                self.img_scale = img_scale
        else:
            if not isinstance(img_scale,Iterable):
                self.img_scale = (img_scale,img_scale)
            else:
                self.img_scale = img_scale

        self.multiscale_mode = multiscale_mode
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx


    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.multiscale_mode:
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            scale, scale_idx = self.img_scale, 0
        
        results['scale'] = (scale[1],scale[0]) #(W,H)
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            img, scale_factor = wmli.resize_imgv2(results[key],
                results['scale'],
                return_scale=True,
                )
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results[key].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h

            results[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        new_shape = results['img'].shape[:2][::-1]
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            masks = results[key].masks
            masks,resized_bboxes = wtu.npresize_mask_in_bboxes(masks,self.old_bboxes,new_shape) 
            results[key] = BitmapMasks(masks)

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            gt_seg = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """
        self.old_bboxes = results[GT_BOXES]
        self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str

@PIPELINES.register_module()
class WMosaic:
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:

         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (height, width).
            Default to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Default to (0.5, 1.5).
        min_bbox_size (int | float): The minimum pixel for filtering
            invalid bboxes after the mosaic pipeline. Default to 0.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` is invalid. Default to True.
        pad_val (int): Pad value. Default to 114.
        prob (float): Probability of applying this transformation.
            Default to 1.0.
    """

    def __init__(self,
                 img_scale=(640, 640), #(H,W)
                 center_ratio_range=(0.5, 1.5),
                 min_bbox_size=0,
                 bbox_clip_border=True,
                 skip_filter=True,
                 pad_val=114,
                 prob=1.0,
                 two_imgs_directions=['horizontal', 'vertical']):
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. '\
            f'got {prob}.'

        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.min_bbox_size = min_bbox_size
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter
        self.pad_val = pad_val
        self.prob = prob
        directions = []
        for sd in two_imgs_directions:
            if sd== "horizontal":
                directions.append(0)
            elif sd == "vertical":
                directions.append(1)
            elif isinstance(sd,int):
                directions.append(sd)
        self.two_imgs_directions = directions

    def __call__(self, results):
        """Call function to make a mosaic of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mosaic transformed.
        """

        results.setdefault('process',[])
        if random.uniform(0, 1) > self.prob:
            results['process'].append(type(self).__name__)
            return results

        if len(results['mix_results'])==3:
            results = self._mosaic_transform_4imgs(results)
        else:
            funcs = [self._mosaic_transform_2imgsh,self._mosaic_transform_2imgsv]
            t = random.choice(self.two_imgs_directions)
            return funcs[t](results)
        results['process'].append(type(self).__name__)
        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        nr = random.choice([1,3])
        indexes = [random.randint(0, len(dataset)-1) for _ in range(nr)]
        return indexes

    def _mosaic_transform_4imgs(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        mosaic_labels = []
        mosaic_bboxes = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), results['img'].shape[-1]),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)
        
        if 'gt_masks' in results:
            mosaic_mask = []
            mosaic_mask_shape = (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2))
        else:
            mosaic_mask = None

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = wmli.resize_img(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
            
            if mosaic_mask is not None:
                mask_i = results_patch['gt_masks'].masks
                gt_bboxes_i = results_patch[GT_BOXES]
                mask_i,_ =  wtu.npresize_mask_in_bboxes(mask_i, gt_bboxes_i,(int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
                n_mask_i = np.zeros([mask_i.shape[0],mosaic_mask_shape[0],mosaic_mask_shape[1]],dtype=np.uint8)

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]
            if mosaic_mask is not None:
                n_mask_i[:,y1_p:y2_p, x1_p:x2_p] = mask_i[:,y1_c:y2_c, x1_c:x2_c]


            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_labels_i = results_patch['gt_labels']

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)
            if mosaic_mask is not None:
                mosaic_mask.append(n_mask_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            if mosaic_mask is not None:
                mosaic_mask = np.concatenate(mosaic_mask, 0).astype(np.uint8)

            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0,
                                                 2 * self.img_scale[1])
                mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0,
                                                 2 * self.img_scale[0])

            if not self.skip_filter:
                mosaic_bboxes, mosaic_labels,valid_inds = \
                    self._filter_box_candidates(mosaic_bboxes, mosaic_labels)
                if mosaic_mask is not None:
                    mosaic_mask = mosaic_mask[valid_inds]

        # remove outside bboxes
        inside_inds = find_inside_bboxes(mosaic_bboxes, 2 * self.img_scale[0],
                                         2 * self.img_scale[1])
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_labels = mosaic_labels[inside_inds]
        if mosaic_mask is not None:
            mosaic_mask = mosaic_mask[inside_inds]
            results['gt_masks'] = BitmapMasks(mosaic_mask)

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels

        return results

    def _mosaic_transform_2imgsh(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        mosaic_labels = []
        mosaic_bboxes = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0]), int(self.img_scale[1] * 2), results['img'].shape[-1]),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0]), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)
        
        if 'gt_masks' in results:
            mosaic_mask = []
            mosaic_mask_shape = (int(self.img_scale[0]), int(self.img_scale[1] * 2))
        else:
            mosaic_mask = None

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(self.img_scale[0]-1)
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = wmli.resize_img(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
            
            if mosaic_mask is not None:
                mask_i = results_patch['gt_masks'].masks
                gt_bboxes_i = results_patch[GT_BOXES]
                mask_i,_ =  wtu.npresize_mask_in_bboxes(mask_i, gt_bboxes_i,(int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
                n_mask_i = np.zeros([mask_i.shape[0],mosaic_mask_shape[0],mosaic_mask_shape[1]],dtype=np.uint8)

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]
            if mosaic_mask is not None:
                n_mask_i[:,y1_p:y2_p, x1_p:x2_p] = mask_i[:,y1_c:y2_c, x1_c:x2_c]


            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_labels_i = results_patch['gt_labels']

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)
            if mosaic_mask is not None:
                mosaic_mask.append(n_mask_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            if mosaic_mask is not None:
                mosaic_mask = np.concatenate(mosaic_mask, 0).astype(np.uint8)

            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0,
                                                 2 * self.img_scale[1])
                mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0,
                                                 2 * self.img_scale[0])

            if not self.skip_filter:
                mosaic_bboxes, mosaic_labels,valid_inds = \
                    self._filter_box_candidates(mosaic_bboxes, mosaic_labels)
                if mosaic_mask is not None:
                    mosaic_mask = mosaic_mask[valid_inds]

        # remove outside bboxes
        inside_inds = find_inside_bboxes(mosaic_bboxes, 2 * self.img_scale[0],
                                         2 * self.img_scale[1])
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_labels = mosaic_labels[inside_inds]
        if mosaic_mask is not None:
            mosaic_mask = mosaic_mask[inside_inds]
            results['gt_masks'] = BitmapMasks(mosaic_mask)

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels

        return results

    def _mosaic_transform_2imgsv(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        mosaic_labels = []
        mosaic_bboxes = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1]), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1])),
                self.pad_val,
                dtype=results['img'].dtype)
        
        if 'gt_masks' in results:
            mosaic_mask = []
            mosaic_mask_shape = (int(self.img_scale[0] * 2), int(self.img_scale[1]))
        else:
            mosaic_mask = None

        # mosaic center x, y
        center_x = int(self.img_scale[1]-1)
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'bottom_left')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = wmli.resize_img(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
            
            if mosaic_mask is not None:
                mask_i = results_patch['gt_masks'].masks
                gt_bboxes_i = results_patch[GT_BOXES]
                mask_i,_ =  wtu.npresize_mask_in_bboxes(mask_i, gt_bboxes_i,(int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
                n_mask_i = np.zeros([mask_i.shape[0],mosaic_mask_shape[0],mosaic_mask_shape[1]],dtype=np.uint8)

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]
            if mosaic_mask is not None:
                n_mask_i[:,y1_p:y2_p, x1_p:x2_p] = mask_i[:,y1_c:y2_c, x1_c:x2_c]


            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_labels_i = results_patch['gt_labels']

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)
            if mosaic_mask is not None:
                mosaic_mask.append(n_mask_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            if mosaic_mask is not None:
                mosaic_mask = np.concatenate(mosaic_mask, 0).astype(np.uint8)

            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0,
                                                 2 * self.img_scale[1])
                mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0,
                                                 2 * self.img_scale[0])

            if not self.skip_filter:
                mosaic_bboxes, mosaic_labels,valid_inds = \
                    self._filter_box_candidates(mosaic_bboxes, mosaic_labels)
                if mosaic_mask is not None:
                    mosaic_mask = mosaic_mask[valid_inds]

        # remove outside bboxes
        inside_inds = find_inside_bboxes(mosaic_bboxes, 2 * self.img_scale[0],
                                         2 * self.img_scale[1])
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_labels = mosaic_labels[inside_inds]
        if mosaic_mask is not None:
            mosaic_mask = mosaic_mask[inside_inds]
            results['gt_masks'] = BitmapMasks(mosaic_mask)

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels

        return results

    def _mosaic_combine(self, loc, center_position_xy, img_shape_wh):
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def _filter_box_candidates(self, bboxes, labels):
        """Filter out bboxes too small after Mosaic."""
        bbox_w = bboxes[:, 2] - bboxes[:, 0]
        bbox_h = bboxes[:, 3] - bboxes[:, 1]
        valid_inds = (bbox_w > self.min_bbox_size) & \
                     (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], labels[valid_inds],valid_inds

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'min_bbox_size={self.min_bbox_size}, '
        repr_str += f'skip_filter={self.skip_filter})'
        return repr_str

@PIPELINES.register_module()
class WGetBBoxesByMask:
    '''
    '''

    def __init__(self,
                 min_bbox_area=4):
        self.min_bbox_area = min_bbox_area


    def _get_bbox_by_mask(self, results):
        masks = results['gt_masks'].masks
        if len(masks) == 0:
            return results
        gtbboxes = []
        for i in range(masks.shape[0]):
            cur_mask = masks[i]
            idx = np.nonzero(cur_mask)
            xs = idx[1]
            ys = idx[0]
            if len(xs)==0:
                gtbboxes.append(np.zeros([4],dtype=np.float32))
            else:
                x0 = np.min(xs)
                y0 = np.min(ys)
                x1 = np.max(xs)
                y1 = np.max(ys)
                gtbboxes.append(np.array([x0,y0,x1,y1],dtype=np.float32))
        
        gtbboxes = np.array(gtbboxes)
        bboxes_area = odb.area(gtbboxes)
        keep = bboxes_area>self.min_bbox_area
        masks = masks[keep]
        gtbboxes = gtbboxes[keep]
        gtlabels = results[GT_LABELS][keep]
        results[GT_BOXES] = gtbboxes 
        results[GT_LABELS] = gtlabels
        results[GT_MASKS] = BitmapMasks(masks)

        return results

    def __call__(self, results):
        return self._get_bbox_by_mask(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_bbox_area={self.min_bbox_area}, '
        return repr_str

@PIPELINES.register_module()
class WGradAguImg:
    def __init__(self):
        pass


    def img_fft(img):
        fft = np.fft.fft2(img)
        fftshift = np.fft.fftshift(fft)
        width = fftshift.shape[1]
        height = fftshift.shape[0]
        dh = height*3//8
        dw = width*3//8
        fftshift[:dh] = 0
        fftshift[-dh:] = 0
        fftshift[:,:dw] = 0
        fftshift[:,-dw:] = 0
    
        fft = np.fft.ifftshift(fftshift)
        n_img = np.fft.ifft2(fft)
        n_img = np.abs(n_img)
        n_img = np.clip(n_img,0,255).astype(np.uint8)
        return n_img

    @staticmethod
    def agu_img(img):
        img = WGradAguImg.img_fft(img)
        grad_x = cv2.Sobel(img,cv2.CV_32F,1,0)
        grad_y = cv2.Sobel(img,cv2.CV_32F,0,1)
        grad_x = cv2.convertScaleAbs(grad_x)
        grad_y = cv2.convertScaleAbs(grad_y)
        grad_xy = cv2.addWeighted(grad_x,0.5,grad_y,0.5,0)
        dilate_kernel = np.ones((15,15),dtype=np.float32)
        grad_xy = cv2.dilate(grad_xy,dilate_kernel,1)
        img = wmli.normal_image(grad_xy)
        return img


    @staticmethod
    def apply(img):
        gray_img = wmli.nprgb_to_gray(img)
        grad_img = WGradAguImg.agu_img(gray_img)
        n_img = np.stack([gray_img,gray_img,grad_img],axis=-1)
        return n_img

    def __call__(self,results):
        img = results['img']
        results['img'] = WGradAguImg.apply(img)

        return results

@PIPELINES.register_module()
class WFFTSmooth:
    def __init__(self):
        pass


    @staticmethod
    def fft_smooth(img):
        fft = np.fft.fft2(img)
        fftshift = np.fft.fftshift(fft)
        width = fftshift.shape[1]
        height = fftshift.shape[0]
        dh = height*3//8
        dw = width*3//8
        fftshift[:dh] = 0
        fftshift[-dh:] = 0
        fftshift[:,:dw] = 0
        fftshift[:,-dw:] = 0
    
        fft = np.fft.ifftshift(fftshift)
        n_img = np.fft.ifft2(fft)
        n_img = np.abs(n_img)
        n_img = np.clip(n_img,0,255).astype(np.uint8)
        return n_img


    @staticmethod
    def apply(img):
        gray_img = wmli.nprgb_to_gray(img)
        smooth_img = WFFTSmooth.fft_smooth(gray_img)
        n_img = np.stack([smooth_img,smooth_img,smooth_img],axis=-1)
        return n_img

    def __call__(self,results):
        img = results['img']
        results['img'] = WFFTSmooth.apply(img)

        return results

@PIPELINES.register_module()
class W2Gray:
    def __init__(self):
        pass

    @staticmethod
    def apply(img):
        gray_img = wmli.nprgb_to_gray(img)
        gray_img = np.expand_dims(gray_img,axis=-1)
        return gray_img

    def __call__(self,results):
        img = results['img']
        results['img'] = W2Gray.apply(img)

        return results

@PIPELINES.register_module()
class WCutOut:
    """CutOut operation.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.

    Args:
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
            shape of dropped regions. It can be `tuple[int, int]` to use a
            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
            shape from the list.
        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
    """

    def __init__(self,
                 prob=0.5,
                 n_holes=[1,5],
                 cutout_size_range=None,
                 cutout_shape=None,
                 cutout_ratio=None,
                 fill_in=(127, 127, 127)):

        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        self.prob = prob
        self.n_holes = n_holes
        self.fill_in = fill_in
        if cutout_size_range is not None:
            assert cutout_shape is None, f"ERROR: can't set cutout_shape and cutout_size range at the same time."
            cutout_shape = []
            for s in range(cutout_size_range[0],cutout_size_range[1]+1):
                cutout_shape.append((s,s))

        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]

    def __call__(self, results):
        """Call function to drop some regions of image."""
        if np.random.rand()>self.prob:
            return results
        h, w, c = results['img'].shape
        n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        for _ in range(n_holes):
            x1 = np.random.randint(0, w-1)
            y1 = np.random.randint(0, h-1)
            index = np.random.randint(0, len(self.candidates))
            if not self.with_ratio:
                cutout_w, cutout_h = self.candidates[index]
            else:
                cutout_w = int(self.candidates[index][0] * w)
                cutout_h = int(self.candidates[index][1] * h)

            x2 = np.clip(x1 + cutout_w, 0, w)
            y2 = np.clip(y1 + cutout_h, 0, h)
            results['img'][y1:y2, x1:x2, :] = self.fill_in

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(n_holes={self.n_holes}, '
        repr_str += (f'cutout_ratio={self.candidates}, ' if self.with_ratio
                     else f'cutout_shape={self.candidates}, ')
        repr_str += f'fill_in={self.fill_in})'
        return repr_str


@PIPELINES.register_module()
class WFixData:
    def __init__(self) -> None:
        pass

    def __call__(self,results):
        if len(results['img'].shape) == 2:
            results['img'] = np.expand_dims(results['img'],axis=-1)
            results['img_shape'] = results['img'].shape
      

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class WCompressMask:
    def __init__(self,allow_overlap=False) -> None:
        '''
        allow_overlap: 是否允许mask重叠，如果允许，那么uint8最多编码8个mask,否则最多可编码255个
        '''
        self.allow_overlap = allow_overlap

    @staticmethod
    def encode(mask,allow_overlap):
        '''
        mask: [N,H,W]
        '''
        if allow_overlap:
            if mask.shape[0]<=1 or mask.shape[0]>=8:
                return mask
            res_mask = np.zeros([1,mask.shape[1],mask.shape[2]],dtype=np.uint8)
            cur_v = 1
            for i in range(mask.shape[0]):
                cur_mask = (np.expand_dims(mask[i],axis=0)*cur_v).astype(np.uint8)
                res_mask += cur_mask
                cur_v *= 2
            return res_mask
        else:
            if mask.shape[0]<=1 or mask.shape[0]>=256:
                return mask
            res_mask = np.zeros([1,mask.shape[1],mask.shape[2]],dtype=np.uint8)
            for i in range(mask.shape[0]):
                cur_mask = (np.expand_dims(mask[i],axis=0)*(i+1)).astype(np.uint8)
                res_mask = np.where(cur_mask>0,cur_mask,res_mask)
            return res_mask
    
    @staticmethod
    def decode(mask,nr,allow_overlap):
        '''
        mask: [1,H,W]
        '''
        if nr<=1 or mask.shape[0]>1:
            return mask
        res_mask = np.zeros([nr,mask.shape[1],mask.shape[2]],dtype=np.uint8)
        if allow_overlap:
            cur_v = 1
            for i in range(nr):
                t_m = (np.ones_like(mask)*cur_v).astype(np.uint8)
                cur_mask  = (mask&t_m).astype(np.uint8)
                res_mask[i] = cur_mask[0]
                cur_v *= 2
        else:
            for i in range(nr):
                t_m = np.ones_like(mask)*(i+1)
                cur_mask = (mask==t_m).astype(np.uint8)
                res_mask[i] = cur_mask[0]
        
        return res_mask

    def __call__(self,results):
        if GT_MASKS in results:
            masks = results[GT_MASKS].masks
            if masks.shape[0]<=1:
                return results
            masks = self.encode(masks,self.allow_overlap)
            results[GT_MASKS] = BitmapMasks(masks)
        
        return results
            
@PIPELINES.register_module()
class W2PolygonMask:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_bboxes_by_contours(contours):
        if len(contours)==0:
            return np.zeros([4],dtype=np.float32)
        cn0 = np.reshape(contours[0],[-1,2])
        x0 = np.min(cn0[:,0])
        x1 = np.max(cn0[:,0])
        y0 = np.min(cn0[:,1])
        y1 = np.max(cn0[:,1])
        for cn in contours[1:]:
            cn = np.reshape(cn,[-1,2])
            x0 = min(np.min(cn[:,0]),x0)
            x1 = max(np.max(cn[:,0]),x1)
            y0 = min(np.min(cn[:,1]),y0)
            y1 = max(np.max(cn[:,1]),y1)
        
        return np.array([x0,y0,x1,y1],dtype=np.float32)

    @staticmethod
    def encode(mask,bboxes):
        if mask.dtype != np.uint8:
            print("ERROR")
        t_masks = []
        keep = np.zeros([mask.shape[0]],dtype=np.bool)
        res_bboxes = []
        for i in range(mask.shape[0]):
            #contours,hierarchy = cv2.findContours(mask[i],cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            contours = wtu.find_contours_in_bbox(mask[i],bboxes[i])
            if len(contours)>0:
                t_masks.append(contours)
                keep[i] = True
                res_bboxes.append(W2PolygonMask.get_bboxes_by_contours(contours))
        if len(res_bboxes) == 0:
            res_bboxes = np.zeros([0,4],dtype=np.float32)
        else:
            res_bboxes = np.array(res_bboxes,dtype=np.float32)
        return PolygonMasks(t_masks,mask.shape[1],mask.shape[2]),res_bboxes,keep

    
    @staticmethod
    def decode(mask,nr):
        '''
        mask: [1,H,W]
        '''
        r_mask = mask.masks
        res_mask = np.zeros([nr,mask.height,mask.width],dtype=np.uint8)
        for i in range(nr):
            res_mask[i] = cv2.drawContours(res_mask[i],mask.masks[i],-1,color=(1,),thickness=cv2.FILLED)

        return BitmapMasks(res_mask,mask.height,mask.width)

    def __call__(self,results):
        if GT_MASKS in results:
            masks = results[GT_MASKS].masks
            masks,bboxes,keep = self.encode(masks,results[GT_BOXES])
            if GT_BOXES in results:
                results[GT_BOXES] = bboxes
            if GT_LABELS in results:
                results[GT_LABELS] = results[GT_LABELS][keep.tolist()]
            results[GT_MASKS] = masks
        
        return results
