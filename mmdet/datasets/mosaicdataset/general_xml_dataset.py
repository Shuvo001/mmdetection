import os
import os.path as osp
import pickle
import xml.etree.ElementTree as ET
from loguru import logger
import glob
import cv2
import numpy as np
import wml_utils as wmlu
import random
from .datasets_wrapper import Dataset
from object_detection2.data_process_toolkit import motion_blur
import sys
from .xml_base_dataset import XmlBaseDataset
from itertools import count

class GeneralXmlDataset(XmlBaseDataset):
    NUM_CLASSES = None
    CLASSES = None
    def __init__(self,
        img_size=(896, 544), # (H,W)
        is_train=True,
        preproc=None,
        cache=True,
        data_dirs = None, # data_dir or list[data_dir] or dict(k=data_dir,v=repeat_nr)
        classes = None, # tuple of data names
        classes_to_id = None, # None or classes name to label, (label idx begin with 0)
        always_make_generate_cache_files=False,
        cache_imgs=True,
        allow_empty_annotation=False,
    ):
        GeneralXmlDataset.CLASSES = classes
        if classes_to_id is None:
            classes_to_id = dict(zip(classes,count()))

        super().__init__(class_to_ind=classes_to_id,
            classes=classes,
            img_size=img_size,
            dataset_name="")
        self.preproc = preproc
        self.img_files = []
        self.xml_files = []
        name = "train" if is_train else "val"
        print(f"Data dirs is {data_dirs}")
        if isinstance(data_dirs,str):
            sub_dirs = {data_dirs:1}
        elif isinstance(data_dirs,(list,tuple)):
            sub_dirs = {}
            for d in data_dirs:
                sub_dirs[d] = 1
        elif isinstance(data_dirs,dict):
            sub_dirs = data_dirs
        else:
            error = f"ERROR: error type of data_dirs type {type(data_dirs)}"
            print(error)
            raise RuntimeError(error)

        for k,v in sub_dirs.items():
            _t_files = wmlu.recurse_get_filepath_in_dir(k,suffix=".bmp;;.jpeg")
            t_files = []
            if not allow_empty_annotation:
                for f in _t_files:
                    txml = wmlu.change_suffix(f,"xml")
                    if osp.exists(txml):
                        t_files.append(f)
            else:
                t_files = _t_files
            print(f"Find {len(t_files)} in dir {k}.")
            if v>1:
                t_files = t_files*v
                print(f"Expand files in {k} to {len(t_files)}.")
            self.img_files.extend(t_files)

        if is_train:
            print(f"Shuffle files.")
            random.seed(73)
            random.shuffle(self.img_files)
            #print(f"Debug.")
            #self.img_files = self.img_files[:100]
            #cache = True
        else:
            cache = False

        for x in self.img_files:
            self.xml_files.append(wmlu.change_suffix(x,"xml"))
        print(f"Total find {len(self.img_files)} {name} files.")

        self.imgs = None
        self.cache_imgs = cache_imgs
        self.always_make_generate_cache_files = always_make_generate_cache_files
        if cache:
            self._cache_images()
        else:
            self.annotations = self._load_coco_annotations()