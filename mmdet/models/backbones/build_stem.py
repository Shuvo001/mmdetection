from thirdparty.registry import Registry

STEM_MODELS = Registry('stem')

def build_stem(type,*args,**kwargs):
    return STEM_MODELS.get(type)(*args,**kwargs)