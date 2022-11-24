from .general_xml_dataset import GeneralXmlDataset
import wml_utils as wmlu
import random
import numpy as np
import torch
import object_detection2.bboxes as odb
from .data_augment import TrainTransform
from .mosaicdetection import MosaicDetection
from ..builder import DATASETS
from iotoolkit.pascal_voc_toolkit import split_voc_files
from itertools import count
import time


@DATASETS.register_module()
class MosaicDetectionDataset(MosaicDetection):
    dataset = None
    def __init__(self,data_dirs,classes,batch_size,img_suffix=".jpg",mean=None,std=None,to_rgb=False,
        name="MosaicData",
        img_size=(1024,1024), #(H,W)
        allow_empty_annotation=True,
        ):
        self.seed = int(time.time())
        class_to_ind = dict(zip(classes,count()))
        self.CLASSES = classes
        preproc=TrainTransform(max_labels=50,
                               flip_prob=0.5,
                               hsv_prob=0.4,
                               mean=mean,
                               std=std,
                               to_rgb=to_rgb)
        self.dataset = GeneralXmlDataset(classes_to_id=class_to_ind,
                                         classes=classes,
                                         img_size=img_size,
                                         preproc=preproc,
                                         data_dirs=data_dirs,
                                         allow_empty_annotation=allow_empty_annotation)

        super().__init__(self.dataset,mosaic=True,
        img_size=img_size,
        preproc=preproc,
        degrees=10.0,
        translate=0.1,
        mosaic_scale=(0.5,1.5),
        mixup_scale=(0.5,1.5),
        shear=2.0,
        perspective=0.0,
        enable_mixup=True,
        mosaic_prob=0.5,
        mixup_prob=0.5,
        perspective_prob=0.5
        )
        self.batch_size = batch_size
