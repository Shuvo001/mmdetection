from thirdparty.registry import Registry
from .build_lr_scheduler import *
from .build_optimizer import *

MODEL_REGISTER = Registry("MODEL")
LOSS_REGISTER = Registry("LOSS")
DATASET_REGISTER = Registry("DATASET")
DATAPROCESSOR_REGISTRY = Registry("DATAPROCESSOR")


'''def build_model(name,cfg, is_train, **kwargs):
    model = MODEL_REGISTER.get(name)(cfg, is_train=is_train,**kwargs)
    return model'''

'''def build_loss(name,*wargs,**kwargs):
    return LOSS_REGISTER.get(name)(*wargs,**kwargs)

def build_dataset(name,*wargs,**kwargs):
    return DATASET_REGISTER.get(name)(*wargs,**kwargs)'''

def build_lr_scheduler(name,kwargs):
    return LR_SCHEDULER_REGISTER.get(name)(**kwargs)

def build_optimizer(cfg, model):
    args = cfg.optimizer
    name = args.pop("type")
    optimizer = OPTIMIZER_REGISTER.get(name)(
            model.parameters(),
            **args
        )
    return optimizer