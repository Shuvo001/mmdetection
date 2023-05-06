from mmcv.runner.hooks import HOOKS, Hook
import wtorch.utils as wtu
import wtorch.train_toolkit as ttu


@HOOKS.register_module()
class WTrainAllParameters(Hook):
    def __init__(self,
                 step=None,
                 lr=None,
                 train_bn=True,
                 ):
        assert step is not None, "step can't be None"
        assert step != 0, "step can't be zero"
        self.train_bn = train_bn
        self.step = step
        self.lr = lr
    
    def before_run(self, trainer):
        if self.step <0:
            self.step = trainer.max_iters+self.step
            print(f"Update to {self}")

    def before_iter(self, runner):
        iter = runner.iter
        if iter < self.step or self.step<=0:
            return
        model = runner.model
        model = wtu.get_model(model)
        if iter >= self.step:
            ttu.defrost_model(model,defrost_bn=self.train_bn)
        args = dict(runner.cfg.optimizer)
        if self.lr is not None:
            args['lr'] = self.lr
        lr = self.lr if self.lr is not None else args['lr']
        wd = args.get('weight_decay',0.0)
        if len(runner.unweights) > 0:
            runner.optimizer.add_param_group({"params": runner.unweights,"weight_decay":wd,"lr":lr})
        if len(runner.unbiases) > 0:
            runner.optimizer.add_param_group({"params": runner.unbiases,"weight_decay":0.0,"lr":lr})
        if len(runner.unbn_weights) > 0:
            runner.optimizer.add_param_group({"params": runner.unbn_weights,"weight_decay":0.0,"lr":lr})
        self.init_lr_scheduler(runner.lr_scheduler,runner.optimizer)
        self.step = 0
        print(f"Train all parameters.")
        

    def __repr__(self):
        return f"{type(self).__name__}: step={self.step}, lr={self.lr}" 
    
    def init_lr_scheduler(self, scheduler,optimizer):
    
        # Attach optimizer
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        scheduler.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]