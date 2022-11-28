import os
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class WCloseMosaic:
    """Class to mosaic.

    Args:
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
    """

    def __init__(self,
                 close_iter_or_epoch =100,
                 by_epoch=False,
                 ):
        self.closed = False
        self.by_epoch = by_epoch
        self.close_iter_or_epoch  = close_iter_or_epoch
    
    def before_run(self, trainer):
        if self.close_iter_or_epoch<0:
            self.close_iter_or_epoch = trainer.max_iters+self.close_iter_or_epoch
            print(f"Update to {self}")
    
    def close_mosaic(self,value,runner):
        if value<=self.close_iter_or_epoch or self.closed:
            return
        if hasattr(runner.data_loader,"close_mosaic"):
            for i in range(10):
                print(f"PID {os.getpid()}: Close mosaic at {value}")
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
        return runner.iter + 1
    
    def __repr__(self):
        return f"{type(self).__name__}: close_iter_or_epoch={self.close_iter_or_epoch}, by_epoch={self.by_epoch}" 
