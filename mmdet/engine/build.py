from thirdparty.registry import Registry
from .build_lr_scheduler import *
from .build_optimizer import *
from wtorch.train_toolkit import simple_split_parameters

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
    bn_weights,weights,biases = simple_split_parameters(model)
    optimizer = OPTIMIZER_REGISTER.get(name)(
            weights,
            **args
        )
    if len(bn_weights)>0:
        optimizer.add_param_group({"params": bn_weights})
    if len(biases)>0:
        optimizer.add_param_group({"params": biases})
    return optimizer