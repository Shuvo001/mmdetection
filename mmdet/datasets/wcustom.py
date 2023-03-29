# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
import copy
import mmcv
import numpy as np
from mmcv.utils import print_log
from torch.utils.data import Dataset
import object_detection2.bboxes as odb
from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose
import object_detection_tools.statistics_tools as st
import pickle
import os
import sys
import time

class WCustomDataset(Dataset):
    """Custom dataset for detection.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

    .. code-block:: none

        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests.
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 ann_file=None,
                 pipeline=None,
                 pipeline2=None,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 cache_processed_data=False,
                 cache_data_items=True,
                 name="dataset"):
        self.name = name
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = classes

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)

        # load annotations (and proposals)
        self._inner_dataset = self.load_annotations(self.ann_file)
        
        #
        statics = st.statistics_boxes_with_datas(self._inner_dataset,
                                          label_encoder=st.default_encode_label,
                                          labels_to_remove=None,
                                          max_aspect=None,absolute_size=True)

        # filter images too small and containing no annotations
        if not test_mode:
            # set group flag for the sampler
            self._set_group_flag()


        if not cache_processed_data and pipeline2 is not None:
            print(f"Auto merge pipeline and pipline2")
            pipeline = pipeline+pipeline2
            pipeline2 = None
        # processing pipeline
        self.pipeline = Compose(pipeline)
        self.pipeline2 = Compose(pipeline2) if pipeline2 is not None else None

        self._data_cache = None
        self._processed_data_cache = None
        if cache_processed_data:
            cache_file_path = self.get_local_cache_file_path(self.ann_file)
            try:
                if osp.exists(cache_file_path):
                    print(f"Load data from cache file {cache_file_path}")
                    with open(cache_file_path,"rb") as f:
                        self._processed_data_cache = pickle.load(f)
                else:
                    self.apply_process_cache(cache_file_path)
            except:
                self.apply_process_cache(cache_file_path)
            
        elif cache_data_items:
            self._data_cache = []
            print("Cache data items")
            sys.stdout.flush()
            for i in range(len(self._inner_dataset)):
                self._data_cache.append(self._inner_dataset[i])
                if i%100 == 0:
                    sys.stdout.write(f"cache {i}/{len(self._inner_dataset)} \r")
                    sys.stdout.flush()
            print(f"Total cache {len(self._data_cache)} data items.")
            sys.stdout.flush()
        self.cache_processed_data = cache_processed_data

    def get_local_cache_file_path(self,ann_file):
        cache_name = self.name+f"_{len(self._inner_dataset)}.pk"
        if osp.isfile(ann_file):
            dir_path = osp.dirname(ann_file)
        elif osp.isdir(ann_file):
            dir_path = ann_file
        else:
            print(f"ERROR ann file {ann_file}")
            raise RuntimeError(f"ERROR ann file {ann_file}")
        
        if not osp.exists(dir_path):
            os.makedirs(dir_path)
        
        return osp.join(dir_path,cache_name)

    def apply_process_cache(self,cache_file_path):
        self.cache_processed_data = False  #先设置为False以使用正常流程处理数据
        self._processed_data_cache = []
        print(f"Cache processed data")
        sys.stdout.flush()
        beg_t = time.time()
        for i in range(len(self._inner_dataset)):
            self._processed_data_cache.append(copy.deepcopy(self.__getitem__(i)))
            if i%20 == 0:
                sys.stdout.write(f"cache {i:05d}/{len(self._inner_dataset):05d} \r")
                sys.stdout.flush()
        print(f"Total cache {len(self._processed_data_cache)} processed data items, {(time.time()-beg_t)/len(self._inner_dataset):.3f} sec/item.")
        print(f"Pipeline for cache is {self.pipeline}")
        print(f"Pipeline not cache is {self.pipeline2}")
        sys.stdout.flush()

        with open(cache_file_path,"wb") as f:
            print(f"Save processed data to cache file {cache_file_path}")
            pickle.dump(self._processed_data_cache,f)
        sys.stdout.flush()
        print(f"cache file size {os.stat(cache_file_path).st_size/1e9:.3f} GB")

    def __len__(self):
        """Total number of samples of data."""
        return len(self._inner_dataset)

    def load_annotations(self, ann_file):
        pass

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        cur_data = self._ann_item(idx)
        img_file, img_shape,labels, labels_names, bboxes, masks, *_ = cur_data
        bboxes = odb.npchangexyorder(bboxes)
        labels = np.array(labels).astype(np.int32)
        ann_info = dict(bboxes=bboxes,labels=labels,bitmap_masks=masks)

        return ann_info

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self._inner_dataset[idx]['ann']['labels'].astype(np.int).tolist()

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['img_prefix'] = None

    def _set_group_flag(self):
        return
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self._inner_dataset)):
            img_info = self._inner_dataset.get_ann_info(i)
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        data = self.get_base_item(idx)

        if self.pipeline2 is not None:
            return self.pipeline2(data)
        else:
            return data

    def get_base_item(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        if self.cache_processed_data:
            return copy.deepcopy(self._processed_data_cache[idx])

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _ann_item(self,idx):
        if self._data_cache is not None:
            return self._data_cache[idx]
        else:
            return self._inner_dataset[idx]

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        cur_data = self._ann_item(idx)
        img_file, img_shape,labels, labels_names, bboxes, masks, *_ = cur_data
        bboxes = odb.npchangexyorder(bboxes)
        labels = np.array(labels).astype(np.int32)
        img_info = dict(filename=img_file,height=img_shape[0],width=img_shape[1])
        ann_info = dict(bboxes=bboxes,labels=labels,bitmap_masks=masks)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """
        cur_data = self._ann_item(idx)
        img_file, img_shape,labels, labels_names, bboxes, masks, *_ = cur_data
        bboxes = odb.npchangexyorder(bboxes)
        labels = np.array(labels).astype(np.int32)
        img_info = dict(filename=img_file,height=img_shape[0],width=img_shape[1])
        ann_info = dict(bboxes=bboxes,labels=labels,bitmap_masks=masks)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)


    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def get_cat2imgs(self):
        """Get a dict with class as key and img_ids as values, which will be
        used in :class:`ClassAwareSampler`.

        Returns:
            dict[list]: A dict of per-label image list,
            the item of the dict indicates a label index,
            corresponds to the image index that contains the label.
        """
        if self.CLASSES is None:
            raise ValueError('self.CLASSES can not be None')
        # sort the label index
        cat2imgs = {i: [] for i in range(len(self.CLASSES))}
        for i in range(len(self)):
            cat_ids = set(self.get_cat_ids(i))
            for cat in cat_ids:
                cat2imgs[cat].append(i)
        return cat2imgs

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

