import cv2
import numpy as np
from ..builder import PIPELINES
from object_detection2.standard_names import *
import img_utils as wmli

@PIPELINES.register_module()
class WLoadImageFromFile:
    def __init__(self,rgb=True):
        self.rgb = rgb

    def __call__(self,img):
        if isinstance(img,str):
            img_path = img
            img = cv2.imread(img_path)
            if self.rgb:
                cv2.cvtColor(img,cv2.COLOR_BGR2RGB,img)
        elif isinstance(img,np.ndarray):
            img_path = ""
        else:
            print(f"Unspport img type {type(img).__name__}")
            return None
        
        results = dict(filename=img_path,
                       img=img,
                       img_shape=img.shape,
                       ori_shape=img.shape,
                       img_fields=['img'])
        return results

    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(rgb={self.rgb})"
        return repr_str

@PIPELINES.register_module()
class WGetImg:
    def __init__(self):
        pass

    def __call__(self,results):
        img = results['img']
        img = np.ascontiguousarray(img)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class WEncodeImg:
    def __init__(self):
        pass

    def __call__(self,results):
        img = results['img']
        if img.shape[-1] == 1:
            img = np.squeeze(img,axis=-1)
        results['img'] = wmli.encode_img(img)
        return results

    
    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class WDecodeImg:
    def __init__(self,fmt='rgb'):
        '''
        fmt: rgb/gray
        '''
        self.fmt = fmt

    def __call__(self,results):
        img =  wmli.decode_img(results['img'],fmt=self.fmt)
        if len(img.shape)==2:
            img = np.expand_dims(img,axis=-1)
        results['img'] = img
        return results

    
    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str