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
from mmcv.parallel.data_container import DataContainer

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
                 rgb=True,
                 bboxes_img_nr=1):
        super(WTensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.log_dir = log_dir
        wmlu.create_empty_dir_remove_if(self.log_dir,key_word="tmp")
        print(f"Log dir {self.log_dir}")
        self.mean = mean
        self.std = std
        self.rgb = rgb
        self.bboxes_img_nr = bboxes_img_nr

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

    @staticmethod
    def get_data(data):
        if isinstance(data,DataContainer):
            return data.data[0]
        return data

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        global_step = self.get_iter(runner)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))
        imgs = self.get_data(runner.inputs['img'])
        idxs = list(range(imgs.shape[0]))
        random.shuffle(idxs)
        idxs = idxs[:self.bboxes_img_nr]
        for idx in idxs:
            gt_bboxes = self.get_data(runner.inputs['gt_bboxes'])
            gt_labels = self.get_data(runner.inputs['gt_labels'])
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
            self.writer.add_image(f"input/img_with_bboxes{idx}",img,global_step=global_step,
                    dataformats="HWC")
        model = runner.model
        if hasattr(model,"module"):
            model = model.module
        summary.log_all_variable(self.writer,model,global_step=global_step)
        

    @master_only
    def after_run(self, runner):
        self.writer.close()
    
    def after_train_iter(self, runner):
        if self.by_epoch and self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif not self.by_epoch and self.every_n_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            runner.log_buffer.average(self.interval)
        global_step = self.get_iter(runner)

        if global_step%self.interval == self.interval-1:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()
