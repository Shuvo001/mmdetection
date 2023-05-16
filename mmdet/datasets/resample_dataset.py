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
from mmdet.utils.datadef import is_debug
from itertools import count
from wtorch.dataset_toolkit import DataUnit
import copy

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

'''
没有任何标注的文件，其采样率的关键字用none表示
'''
class WResampleDataset(Dataset):
    CLASSES = None

    PALETTE = None

    def __init__(self,
                 dataset,
                 classes,
                 data_resample_parameters,
                 base_repeat_nr:int=1):
        self._inner_dataset = dataset
        self.CLASSES = classes
        self.base_repeat_nr = int(base_repeat_nr)
        self.idx2idx = self.get_idx2idx(data_resample_parameters)
        if is_debug():
            print(f"Resample idx2idx")
            for i,x in enumerate(self.idx2idx):
                print(i,x)
            print(f"inner dataset info")
            for i in range(len(self._inner_dataset)):
                info = self._inner_dataset.get_ann_info(i)
                print(i,info['filename'],info['labels'])
    
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
                idx2idx.extend(v*self.base_repeat_nr)
                print(f"label {k} default repeat {self.base_repeat_nr} times, total {len(v)*self.base_repeat_nr} samples.")
            else:
                nr = rdata_resample_parameters[k]*self.base_repeat_nr
                print(f"label {k} repeat {nr} times, total {len(v)} samples.")
                if wmlu.is_int(nr):
                    nr = int(nr)
                    idx2idx.extend(list(list(v)*nr))
                elif nr<1.0:
                    numerator,denominator = wmlu.to_fraction(nr)
                    nrs = wmlu.list_to_2dlist(v,size=denominator)
                    for _nrs in nrs:
                        d = DataUnit(_nrs)
                        repeat_nr = int(max(numerator*len(_nrs)/denominator,1))
                        ds = [d]*repeat_nr
                        ds = [copy.deepcopy(x) for x in ds]
                        idx2idx.extend(ds)
                else:
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
        if is_debug():
            old_idx = idx
        idx = self.trans_idx(idx)
        results = self._inner_dataset.__getitem__(idx)
        if is_debug():
            results['filename'] = results['filename']+f"#{old_idx}"
        return results