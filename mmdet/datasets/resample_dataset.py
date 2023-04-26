import os.path as osp
from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset
import object_detection2.bboxes as odb
import datasets_tools.statistics_tools as st
import wml_utils as wmlu
import pickle
import os
import sys
import time
from itertools import count
from wtorch.dataset_toolkit import DataUnit

def get_resample_nr(labels,resample_parameters):
    if len(labels)==0:
        if None in resample_parameters:
            return resample_parameters[None],None
        else:
            return 1
    major_label = -1
    nr = 0
    for l in labels:
        if l in resample_parameters:
            nr = max(nr,resample_parameters[l])
            major_label = l
    if major_label < 0:
        return 1,major_label
    return nr,major_label

class WResampleDataset(Dataset):
    CLASSES = None

    PALETTE = None

    def __init__(self,
                 dataset,
                 classes,
                 data_resample_parameters):
        self._inner_dataset = dataset
        self.CLASSES = classes
        self.idx2idx = self.get_idx2idx(data_resample_parameters)
    
    def get_idx2idx(self,data_resample_parameters):
        if data_resample_parameters is None or len(data_resample_parameters)==0:
            return None
        label_text2id = dict(zip(self.CLASSES,count()))
        label_text2id["none"] = None
        rdata_resample_parameters = {}
        for k,v in data_resample_parameters.items():
            rdata_resample_parameters[label_text2id[k]] = v
        l2idx = wmlu.MDict(dtype=list)
        for i in range(len(self._inner_dataset)):
            info = self._inner_dataset.get_ann_info(i)
            labels = info['labels']
            nr,l = get_resample_nr(labels,rdata_resample_parameters)
            l2idx[l].append(i)
        print(f"get idx2idx")
        idx2idx = []
        for k,v in l2idx.items():
            if k not in rdata_resample_parameters:
                idx2idx.extend(v)
                print(f"label {k} default repeat 1 times, total {len(v)} samples.")
            else:
                nr = rdata_resample_parameters[k]
                print(f"label {k} repeat {nr} times, total {len(v)} samples.")
                nnr = int(nr*len(v))
                d = DataUnit(v)
                idx2idx.extend([d]*nnr)
        print(f"Resample {len(self._inner_dataset)} to {len(idx2idx)}") 
        return idx2idx

    def trans_idx(self,idx):
        d = self.idx2idx[idx]
        if isinstance(d,DataUnit):
            return d.sample()
        else:
            return d

    def __len__(self):
        """Total number of samples of data."""
        return len(self.idx2idx)

    def get_ann_info(self, idx):
        idx = self.trans_idx(idx)
        return self._inner_dataset.get_ann_info(idx)


    def get_cat_ids(self, idx):
        idx = self.trans_idx(idx)
        return self._inner_dataset.get_cat_ids(idx)

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
        idx = self.trans_idx(idx)
        return self._inner_dataset._rand_another(idx)

    def __getitem__(self, idx):
        idx = self.trans_idx(idx)
        return self._inner_dataset.__getitem__(idx)