# Copyright (c) OpenMMLab. All rights reserved.
import re

from mmcv.utils import Config

def replace_key_vals(cfg,key,v):
    for k,v in cfg.items():
        if k == key:
            cfg[k] = v
        elif isinstance(v,dict):
            replace_key_vals(v,key,v)

def replace_cfg_vals(ori_cfg):
    """Replace the string "${key}" with the corresponding value.

    Replace the "${key}" with the value of ori_cfg.key in the config. And
    support replacing the chained ${key}. Such as, replace "${key0.key1}"
    with the value of cfg.key0.key1. Code is modified from `vars.py
    < https://github.com/microsoft/SoftTeacher/blob/main/ssod/utils/vars.py>`_  # noqa: E501

    Args:
        ori_cfg (mmcv.utils.config.Config):
            The origin config with "${key}" generated from a file.

    Returns:
        updated_cfg [mmcv.utils.config.Config]:
            The config with "${key}" replaced by the corresponding value.
    """

    def get_value(cfg, key):
        for k in key.split('.'):
            cfg = cfg[k]
        return cfg

    def replace_value(cfg):
        if isinstance(cfg, dict):
            return {key: replace_value(value) for key, value in cfg.items()}
        elif isinstance(cfg, list):
            return [replace_value(item) for item in cfg]
        elif isinstance(cfg, tuple):
            return tuple([replace_value(item) for item in cfg])
        elif isinstance(cfg, str):
            # the format of string cfg may be:
            # 1) "${key}", which will be replaced with cfg.key directly
            # 2) "xxx${key}xxx" or "xxx${key1}xxx${key2}xxx",
            # which will be replaced with the string of the cfg.key
            keys = pattern_key.findall(cfg)
            values = [get_value(ori_cfg, key[2:-1]) for key in keys]
            if len(keys) == 1 and keys[0] == cfg:
                # the format of string cfg is "${key}"
                cfg = values[0]
            else:
                for key, value in zip(keys, values):
                    # the format of string cfg is
                    # "xxx${key}xxx" or "xxx${key1}xxx${key2}xxx"
                    assert not isinstance(value, (dict, list, tuple)), \
                        f'for the format of string cfg is ' \
                        f"'xxxxx${key}xxxxx' or 'xxx${key}xxx${key}xxx', " \
                        f"the type of the value of '${key}' " \
                        f'can not be dict, list, or tuple' \
                        f'but you input {type(value)} in {cfg}'
                    cfg = cfg.replace(key, str(value))
            return cfg
        else:
            return cfg

    # the pattern of string "${key}"
    pattern_key = re.compile(r'\$\{[a-zA-Z\d_.]*\}')
    # the type of ori_cfg._cfg_dict is mmcv.utils.config.ConfigDict
    updated_cfg = Config(
        replace_value(ori_cfg._cfg_dict), filename=ori_cfg.filename)
    # replace the model with model_wrapper
    if updated_cfg.get('model_wrapper', None) is not None:
        updated_cfg.model = updated_cfg.model_wrapper
        updated_cfg.pop('model_wrapper')
    train_dataset = updated_cfg.get('train_dataset',None)
    if train_dataset is not None:
        updated_cfg.data.train = train_dataset
        updated_cfg.train_dataset = None

    data_root = updated_cfg.get('data_root',None)
    if data_root is not None:
        print(f"Update train data ann_file to {data_root}")
        replace_key_vals(updated_cfg['data'],"ann_file",data_root)

    test_data_dir = updated_cfg.get('test_data_dir',None)
    if updated_cfg.data.val.data_dirs != test_data_dir:
        print(f"Update val data dirs to {test_data_dir}")
        updated_cfg.data.val.data_dirs = test_data_dir
        updated_cfg.test_data_dir = None

    '''train_pipeline = updated_cfg.get('train_pipeline',None)
    if train_pipeline is not None:
        replace_key_vals(updated_cfg['data'].train,"pipeline",train_pipeline)
        updated_cfg.train_pipeline = None'''

    samples_per_gpu = updated_cfg.get('samples_per_gpu',None)
    if samples_per_gpu is not None:
        replace_key_vals(updated_cfg['data'],"samples_per_gpu",samples_per_gpu)

    return updated_cfg
