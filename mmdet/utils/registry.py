from mmcv.utils import Registry as MMRegistry
import copy

class Registry(MMRegistry):
    def build(self,cfg):
        if 'type' not in cfg:
            if 'default' in cfg and 'cfgs' in cfg:
                default = cfg['default']
                cfgs = []
                for lcfg1 in cfg['cfgs']:
                    lcfg = copy.deepcopy(default)
                    lcfg.update(lcfg1)
                    cfgs.append(lcfg)
                return super().build(cfgs)
        return super().build(cfg)

    @staticmethod
    def expand2list(cfg,num):
        if cfg is None:
            return None
        if 'default' in cfg and 'cfgs' in cfg:
            default = cfg['default']
            cfgs = []
            for lcfg1 in cfg['cfgs']:
                lcfg = copy.deepcopy(default)
                lcfg.update(lcfg1)
                cfgs.append(lcfg)
            return cfgs

        if not isinstance(cfg, list):
            cfgs = [
                cfg for _ in range(num)
            ]
            return cfgs
        
        return cfg





