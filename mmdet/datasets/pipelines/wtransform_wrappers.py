from ..builder import PIPELINES
from .compose import Compose
from mmcv.utils import build_from_cfg
import random
import numpy as np


@PIPELINES.register_module()
class WRandomChoice:
    def __init__(self,transforms):
        prob_sum = 0.0
        self.probs = []
        self.transforms = []
        for trans_dict,prob in transforms:
            if isinstance(trans_dict,dict):
                transform = build_from_cfg(trans_dict, PIPELINES)
            elif isinstance(trans_dict,(list,tuple)):
                transform = Compose(trans_dict)
            else:
                print(f"ERROR: unsupport trans data type {type(trans_dict)}, {trans_dict}")
                raise RuntimeError(f"ERROR: unsupport trans data type {type(trans_dict)}, {trans_dict}")
            self.transforms.append(transform)
            self.probs.append(prob)
            prob_sum += prob
        if prob_sum<1.0:
            self.transforms.append(None)
            self.probs.append(1.0-prob_sum)
        elif prob_sum>1.0:
            print(f"ERROR: probs sum greater than 1.0, {prob_sum}, probs={self.probs}")
            raise RuntimeError(f"ERROR: probs sum greater than 1.0, {prob_sum}, probs={self.probs}")

    def __call__(self, results):
        idx = np.random.choice(len(self.transforms),1,p=self.probs)[0]
        transform = self.transforms[idx]
        if transform is None:
            return results
        return transform(results)
