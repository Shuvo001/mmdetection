from .xml_base_dataset import XmlBaseDataset
import wml_utils as wmlu
import random
import numpy as np
import torch
import uuid
from wtorch.data import DataLoader as torchDataLoader
import object_detection2.bboxes as odb
from object_detection2.standard_names import *
from .data_augment import TrainTransform
from .mosaicdetection import MosaicDetection
from ..builder import DATASETS
from .samplers import *
import time

def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)

class DataLoader(torchDataLoader):
    """
    Lightnet dataloader that enables on the fly resizing of the images.
    See :class:`torch.utils.data.DataLoader` for more information on the arguments.
    Check more on the following website:
    https://gitlab.com/EAVISE/lightnet/-/blob/master/lightnet/data/_dataloading.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__initialized = False
        shuffle = False
        batch_sampler = None
        if len(args) > 5:
            shuffle = args[2]
            sampler = args[3]
            batch_sampler = args[4]
        elif len(args) > 4:
            shuffle = args[2]
            sampler = args[3]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        elif len(args) > 3:
            shuffle = args[2]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        else:
            if "shuffle" in kwargs:
                shuffle = kwargs["shuffle"]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]

        # Use custom BatchSampler
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
                    # sampler = torch.utils.data.DistributedSampler(self.dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            batch_sampler = YoloBatchSampler(
                sampler,
                self.batch_size,
                self.drop_last,
                input_dimension=self.dataset.input_dim,
            )
            # batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations =

        self.batch_sampler = batch_sampler

        self.__initialized = True

    def close_mosaic(self):
        self.batch_sampler.mosaic = False


@DATASETS.register_module()
class MosaicDetectionDataset(object):
    dataset = None
    def __init__(self,data_dirs,category_index,batch_size,img_suffix=".jpg",mean=None,std=None,to_rgb=False):
        self.seed = int(time.time())
        img_size = (1024,1024)
        class_to_ind = {}
        classes = []
        for k,v in category_index.items():
            class_to_ind[v] = k
            classes.append(v)
        self.CLASSES = classes
        preproc=TrainTransform(max_labels=50,
                               flip_prob=0.5,
                               hsv_prob=0.4,
                               mean=mean,
                               std=std,
                               to_rgb=to_rgb)
        self.dataset = XmlBaseDataset(class_to_ind=class_to_ind,classes=classes,
                                      img_size=img_size,
                                       preproc=preproc)
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

        self.dataset.img_files = self.img_files
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
        self.dataset.CLASSES = classes

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=True,
        )

        dataloader_kwargs = {"num_workers": 16, "pin_memory": False}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["batch_split_nr"] = 2

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        self.loader = DataLoader(self.dataset, **dataloader_kwargs)