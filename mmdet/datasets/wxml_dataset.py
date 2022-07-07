# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET
import wml_utils as wmlu
import mmcv
import numpy as np
from PIL import Image
from .pipelines import Compose
from .builder import DATASETS
from .custom import CustomDataset
from iotoolkit.pascal_voc_toolkit import read_voc_xml
import object_detection2.bboxes as odb
from torch.utils.data import Dataset
import random


@DATASETS.register_module()
class WXMLDataset(Dataset):
    """XML dataset for detection.

    Args:
        min_size (int | float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field.
        img_subdir (str): Subdir where images are stored. Default: JPEGImages.
        ann_subdir (str): Subdir where annotations are. Default: Annotations.
    """

    def __init__(self,
                 data_dirs=None,
                 img_suffix = ".jpg",
                 min_size=None,
                 pipeline = None,
                 classes=None,
                 test_mode=False,
                 **kwargs):
        super().__init__()
        if isinstance(data_dirs,str):
            data_dirs = [data_dirs]
        self.img_files = [] 
        for dir in data_dirs:
            if not isinstance(dir,str):
                dir,repeat_nr = dir
            else:
                repeat_nr = 0
            _imgs = wmlu.recurse_get_filepath_in_dir(dir,suffix=img_suffix)
            if repeat_nr > 1:
                imgs = _imgs*repeat_nr
                print(f"Find {len(_imgs)} in {dir}, expand to {len(imgs)} imgs.")
            else:
                imgs = _imgs
                print(f"Find {len(imgs)} in {dir}")
            self.img_files.extend(imgs)
        self.test_mode = test_mode
        print(f"Total find {len(self.img_files)} files.")
        self.xml_files = []
        self.CLASSES = classes
        self.classes_name2id = {}
        for i,name in enumerate(self.CLASSES):
            self.classes_name2id[name] = i
        for x in self.img_files:
            self.xml_files.append(wmlu.change_suffix(x,"xml"))
        self.annos = []
        for x in range(len(self.img_files)):
            self.annos.append(self.read_ann_info(x))
        

        self._set_group_flag()

        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.xml_files)
    

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            shape = self.annos[i]['shape']
            if shape[1] / shape[0] > 1:
                self.flag[i] = 1
    
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def _rand_another(self, idx):
        res = random.randint(0,len(self)-2)
        if res==idx:
            return (res+1)%len(self)
        return res

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        filename = self.img_files[idx]
        ann_info = self.get_ann_info(idx)
        results = {}
        self.pre_pipeline(results)
        img_info = {}
        img_info['filename'] = filename
        results['ann_info'] = ann_info
        results['img_info'] = img_info
        results['filename'] = filename
        results['img_shape'] = ann_info['shape']
        results['bbox_fileds'] = []
        return self.pipeline(results) 

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        #results['img_prefix'] = self.img_prefix
        #results['seg_prefix'] = self.seg_prefix
        #results['proposal_file'] = self.proposal_file
        results['img_prefix'] = None
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def get_ann_info(self,idx):
        return self.annos[idx]

    def read_ann_info(self,idx):
        xml_file = self.xml_files[idx]
        shape, bboxes, labels_text, difficult, truncated, probs = read_voc_xml(xml_file,absolute_coord=True)
        bboxes = odb.npchangexyorder(bboxes)
        labels = [self.classes_name2id[x] for x in labels_text]
        labels = np.array(labels,dtype=np.int32)
        ann_info = {"bboxes":bboxes,"labels":labels,'shape':shape}
        return ann_info


    def evaluate(self,
                results,
                metric='mAP',
                logger=None,
                proposal_nums=(100, 300, 1000),
                iou_thr=0.5,
                scale_ranges=None):
       return {}
