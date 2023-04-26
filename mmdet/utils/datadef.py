import os.path as osp

class AnchorFmt:
    AF_CXCYWH = 0
    AF_X0Y0X1Y1 = 1

DEFAULT_WEIGHTS_DIR="/home/wj/ai/work/mmdetection/weights"

def get_weight_path(path):
    if osp.exists(path):
        return path
    t_path = osp.join(DEFAULT_WEIGHTS_DIR,path)
    if osp.exists(t_path):
        return t_path
    return path

__debug = False

def is_debug():
    return __debug

def set_debug(debug=True):
    global __debug
    __debug = debug