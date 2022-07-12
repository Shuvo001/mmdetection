# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook

import wml_utils as wmlu
from wtorch.utils import unnormalize
import random
import numpy as np
import object_detection2.bboxes as odb
import object_detection2.visualization as odv
import wtorch.summary as summary
from mmcv.parallel.data_container import DataContainer

@HOOKS.register_module()
class WCloseMosaic(Hook):
    """Class to log metrics to Tensorboard.

    Args:
        log_dir (string): Save directory location. Default: None. If default
            values are used, directory location is ``runner.work_dir``/tf_logs.
        interval (int): Logging interval (every k iterations). Default: True.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
    """

    def __init__(self,
                 close_iter_or_epoch =100,
                 by_epoch=False,
                 ):
        super().__init__()
        self.closed = False
        self.by_epoch = by_epoch
        self.close_iter_or_epoch  = close_iter_or_epoch
    
    def close_mosaic(self,value,runner):
        if value<=self.close_iter_or_epoch or self.closed:
            return
        for i in range(10):
            print(f"Close mosaic at {value}")
        runner.data_loader.close_mosaic()
        self.closed = True
    
    def before_epoch(self, runner):
        if not self.by_epoch:
            return
        v = runner.epoch
        self.close_mosaic(v,runner)


    def before_iter(self, runner):
        if self.by_epoch:
            return
        v = self.get_iter(runner)
        self.close_mosaic(v,runner)
    
    def get_iter(self, runner, inner_iter=False):
        """Get the current training iteration step."""
        if self.by_epoch and inner_iter:
            current_iter = runner.inner_iter + 1
        else:
            current_iter = runner.iter + 1
        return current_iter
