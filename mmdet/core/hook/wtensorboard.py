# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks import LoggerHook
import wml_utils as wmlu
from wtorch.utils import unnormalize
import random
import numpy as np
import object_detection2.bboxes as odb
import object_detection2.visualization as odv
import wtorch.summary as summary

@HOOKS.register_module()
class WTensorboardLoggerHook(LoggerHook):
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
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True,
                 mean=None,
                 std=None,
                 rgb=True):
        super(WTensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.log_dir = log_dir
        wmlu.create_empty_dir_remove_if(self.log_dir,key_word="tmp")
        print(f"Log dir {self.log_dir}")
        self.mean = mean
        self.std = std
        self.rgb = rgb

    @master_only
    def before_run(self, runner):
        super(WTensorboardLoggerHook, self).before_run(runner)
        if (TORCH_VERSION == 'parrots'
                or digit_version(TORCH_VERSION) < digit_version('1.1')):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        global_step = self.get_iter(runner)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))
        imgs = runner.inputs['img'].data[0]
        idx = random.randint(0,imgs.shape[0]-1)
        gt_bboxes = runner.inputs['gt_bboxes'].data[0]
        gt_labels = runner.inputs['gt_labels'].data[0]
        img = imgs[idx]
        gt_bboxes = gt_bboxes[idx].cpu().numpy()
        gt_labels = gt_labels[idx].cpu().numpy()
        if self.mean is not None:
            img = unnormalize(img,mean=self.mean,std=self.std).cpu().numpy()
        img = np.transpose(img,[1,2,0])
        if not self.rgb:
            img = img[...,::-1]
        gt_bboxes = odb.npchangexyorder(gt_bboxes)
        img = np.ascontiguousarray(img)
        img = odv.draw_bboxes(img,gt_labels,bboxes=gt_bboxes,is_relative_coordinate=False)
        img = np.clip(img,0,255).astype(np.uint8)
        self.writer.add_image("input/img_with_bboxes",img,global_step=global_step,
                dataformats="HWC")
        model = runner.model
        if hasattr(model,"module"):
            model = model.module
        summary.log_all_variable(self.writer,model,global_step=global_step)
        

    @master_only
    def after_run(self, runner):
        self.writer.close()
