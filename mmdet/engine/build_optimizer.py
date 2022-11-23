import torch.optim as optim
from thirdparty.registry import Registry
import wml_utils as wmlu


OPTIMIZER_REGISTER = Registry("optimizer")


for k,v in optim.__dict__.items():
    if wmlu.is_child_of(v, optim.Optimizer):
        OPTIMIZER_REGISTER.register(v)

