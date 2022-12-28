import uuid
import torch
import numpy as np
from .samplers import *
import time
import random
from .default_dataloader import DataLoader
from thirdparty.registry import Registry

DATALOADER_REGISTER = Registry("DATALOADER")

def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)

@DATALOADER_REGISTER.register()
def yolo_dataloader(dataset,cfg,samples_per_gpu,num_workers=16,seed=None,pin_memory=False,dist=True,num_gpus=1,**kwargs):
    sampler = InfiniteSampler(len(dataset), seed=seed if seed is not None else int(time.time()))
    batch_sampler = YoloBatchSampler(
        sampler=sampler,
        batch_size=samples_per_gpu,
        drop_last=False,
        mosaic=True,
    )

    dataloader_kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}
    dataloader_kwargs["batch_sampler"] = batch_sampler
    #dataloader_kwargs["batch_split_nr"] = 2

    # Make sure each process has different random seed, especially for 'fork' method.
    # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
    dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
    dataloader_kwargs.update(kwargs)

    data_loader = DataLoader(dataset, **dataloader_kwargs)

    return data_loader

def build_dataloader(name,dataset,cfg,samples_per_gpu,num_workers=16,seed=None,pin_memory=False,num_gpus=1,dist=True,**kwargs):
    build_func = DATALOADER_REGISTER.get(name)
    return build_func(dataset=dataset,cfg=cfg,samples_per_gpu=samples_per_gpu,num_workers=num_workers,
    seed=seed,
    pin_memory=pin_memory,
    num_gpus=num_gpus,
    dist=dist,**kwargs)