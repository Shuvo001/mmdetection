class AnchorFmt:
    AF_CXCYWH = 0
    AF_X0Y0X1Y1 = 1

__debug = False

def is_debug():
    return __debug

def set_debug(debug=True):
    global __debug
    __debug = debug