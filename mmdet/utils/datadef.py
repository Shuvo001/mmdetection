__debug = False

def is_debug():
    return __debug

def set_debug(debug=True):
    global __debug
    __debug = debug