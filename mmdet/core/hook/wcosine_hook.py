from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks import LrUpdaterHook
from mmcv.runner.hooks.lr_updater import (CosineAnnealingLrUpdaterHook,
                                          annealing_cos)

@HOOKS.register_module()
class WCosineAnnealingLrUpdaterHook(LrUpdaterHook):
    """CosineAnnealing LR scheduler.

    Args:
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.by_epoch = False
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super().__init__(**kwargs)
    
    def get_warmup_lr(self, cur_iters):

        def _get_warmup_lr(cur_iters, regular_lr):
            # exp warmup scheme
            k = self.warmup_ratio * pow(
                (cur_iters + 1) / float(self.warmup_iters), 2)
            warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.base_lr, dict):
            lr_groups = {}
            for key, base_lr in self.base_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, base_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.base_lr)

    def get_lr(self, runner, base_lr):
        last_iter = 0

        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = max(0,runner.max_iters-2000)

        progress += 1

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        if progress >= max_progress - last_iter:
            # fixed learning rate
            return target_lr
        else:
            return annealing_cos(
                base_lr, target_lr, (progress - self.warmup_iters) /
                (max_progress - self.warmup_iters - last_iter))
