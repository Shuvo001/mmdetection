import torch.optim.lr_scheduler as lr_scheduler
import wtorch.wlr_scheduler as wlr_scheduler
import wml_utils as wmlu
from thirdparty.registry import Registry

LR_SCHEDULER_REGISTER = Registry("lr_scheduler")


for k,v in lr_scheduler.__dict__.items():
    if wmlu.is_child_of(v, lr_scheduler._LRScheduler):
        LR_SCHEDULER_REGISTER.register(v)

for k,v in wlr_scheduler.__dict__.items():
    if wmlu.is_child_of(v, wlr_scheduler._LRScheduler):
        LR_SCHEDULER_REGISTER.register(v)