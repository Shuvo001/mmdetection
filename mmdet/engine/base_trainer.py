import mmcv
from mmcv.runner.hooks import HOOKS
import wml_utils as wmlu

'''
class Hook:
    def before_run(self, trainer):
        pass

    def after_run(self, trainer):
        pass

    def before_epoch(self, trainer):
        pass

    def after_epoch(self, trainer):
        pass

    def before_iter(self, trainer):
        pass

    def after_iter(self, trainer):
        pass
'''

hook_func_nams = ["before_run", "after_run", "before_epoch", "after_epoch" "before_iter","after_iter"]

class BaseTrainer:
    def __init__(self,cfg):
        self.cfg = cfg
        self._hooks = []
        if hasattr(self.cfg,"hooks"):
            self.register_custom_hooks(self.cfg.hooks)
        print("HOOKS:")
        wmlu.show_list(self._hooks)
        pass

    def call_hook(self, fn_name):
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            if not hasattr(hook,fn_name):
                continue
            getattr(hook, fn_name)(self)
    
    def register_hook_from_cfg(self, hook_cfg):
        """Register a hook from its cfg.

        Args:
            hook_cfg (dict): Hook config. It should have at least keys 'type'
              and 'priority' indicating its type and priority.

        Note:
            The specific hook class to register should not use 'type' and
            'priority' arguments during initialization.
        """
        hook_cfg = hook_cfg.copy()
        hook = mmcv.build_from_cfg(hook_cfg, HOOKS)
        self.register_hook(hook)
    
    def register_hook(self, hook):
        has_func = False
        for name in hook_func_nams:
            if hasattr(hook,name):
                has_func = True
                break

        if not has_func:
            print(f"ERROR: {hook} has not implemented any function in {hook_func_nams}")
            return

        self._hooks.append(hook)
    
    def register_custom_hooks(self, custom_config):
        if custom_config is None:
            return

        if not isinstance(custom_config, list):
            custom_config = [custom_config]

        for item in custom_config:
            if isinstance(item, dict):
                self.register_hook_from_cfg(item)
            else:
                self.register_hook(item)
