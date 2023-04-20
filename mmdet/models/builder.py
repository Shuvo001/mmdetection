# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch.nn as nn
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
from wtorch.dropblock import DropBlock2D,LinearScheduler
from collections import Iterable
import copy

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS
SECOND_STAGE_HOOKS = MODELS




def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head."""
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_detector(cfg):
    """Build detector."""
    '''
    cfg: 顶层config.model
    train_cfg 默认使用config.model.train_cfg
    test_cfg 默认使用config.model.test_cfg
    '''
    '''
    禁止原实现通过外部输入,train_cfg,test_cfg
    c
    '''
    return DETECTORS.build(cfg)

def build_second_stage_hook(cfg):
    return SECOND_STAGE_HOOKS.build(cfg)

def build_drop_blocks(cfg):
    '''
   cfg:{ "dropout":{"type":"DropBlock2D"},"drop_prob":[0.1,0.1,0.1,0.1],"block_size":[4,3,2,1]},
   "scheduler":{"type":"LinearScheduler","begin_step":5000}}
    '''
    drop_prob = cfg['dropout']['drop_prob']
    block_size = cfg['dropout']['block_size']
    scheduler = copy.deepcopy(cfg['scheduler'])
    stype = scheduler.pop("type","LinearScheduler")
    if stype != "LinearScheduler":
        print(f"{stype} is not support.")
        raise RuntimeError(f"{stype} is not support.")
    nr = len(block_size)

    if not isinstance(drop_prob,Iterable):
        drop_prob = [drop_prob]*nr

    models = []
    for i in range(nr):
        do = DropBlock2D(drop_prob=drop_prob[i],block_size=block_size[i])
        m = LinearScheduler(do,**scheduler)
        models.append(m)

    return nn.ModuleList(models)

