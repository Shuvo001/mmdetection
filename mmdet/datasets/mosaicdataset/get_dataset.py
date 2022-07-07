from .xml_base_dataset import XmlBaseDataset
import wml_utils as wmlu
import random
import numpy as np
import torch
import uuid
from wtorch.data import DataLoader
import object_detection2.bboxes as odb
from object_detection2.standard_names import *
from .data_augment import TrainTransform
from .mosaicdetection import MosaicDetection

def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)

class MosaicDetectionDataset(object):
    dataset = None
    def __init__(self,data_path,category_index,batch_size,total_steps):
        img_size = (1024,1024)
        class_to_ind = {}
        classes = []
        for k,v in category_index.items():
            class_to_ind[v] = k
            classes.append(v)
        preproc=TrainTransform(max_labels=50,
                               flip_prob=0.5,
                               hsv_prob=0.4)
        self.dataset = XmlBaseDataset(class_to_ind=class_to_ind,classes=classes,
                                      img_size=img_size,
                                       preproc=preproc)
        self.dataset.img_files = wmlu.recurse_get_filepath_in_dir(data_path,suffix=".jpg;;.bmp")
        self.dataset.xml_files = [wmlu.change_suffix(x,"xml") for x in self.dataset.img_files]
        self.dataset._cache_images()
        self.dataset = MosaicDetection(self.dataset,mosaic=True,
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
        total_nr = batch_size*total_steps
        self.idxs = []
        while len(self.idxs)<total_nr:
            idxs = list(range(len(self.dataset)))
            random.shuffle(idxs)
            self.idxs.extend(idxs)
        self.idxs = np.array(self.idxs[:total_nr],dtype=np.int32)
        self._idxs = np.reshape(self.idxs,[-1,batch_size])
        self.idxs = []
        threshold = self._idxs.shape[0]*0.8
        for i,x in enumerate(self._idxs):
            item = []
            enable = i<threshold
            for v in x:
                item.append((enable,v))
            self.idxs.append(item)

        dataloader_kwargs = {"num_workers": 16, "pin_memory": False}
        dataloader_kwargs["batch_sampler"] = self.idxs
        dataloader_kwargs["batch_split_nr"] = 2

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        self.loader = DataLoader(self.dataset, **dataloader_kwargs)