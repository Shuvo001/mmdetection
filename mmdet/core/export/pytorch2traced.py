import img_utils as wmli
import wtorch.utils as wtu
import torch
import numpy as np


def preprocess_2traced_example_input(input_config):
    """Prepare an example input image for ``generate_inputs_and_wrap_model``.

    Args:
        input_config (dict): customized config describing the example input.

    Returns:
        tuple: (one_img, one_meta), tensor of the example input image and \
            meta information for the example input image.

    Examples:
        >>> from mmdet.core.export import preprocess_example_input
        >>> input_config = {
        >>>         'input_shape': (1,3,224,224),   #(B,C,H,W)
        >>>         'input_path': 'demo/demo.jpg',
        >>>         'normalize_cfg': {
        >>>             'mean': (123.675, 116.28, 103.53),
        >>>             'std': (58.395, 57.12, 57.375)
        >>>             }
        >>>         }
        >>> one_img, one_meta = preprocess_example_input(input_config)
        >>> print(one_img.shape)
        torch.Size([1, 3, 224, 224])
        >>> print(one_meta)
        {'img_shape': (224, 224, 3),
        'ori_shape': (224, 224, 3),
        'pad_shape': (224, 224, 3),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False}
    """
    input_path = input_config['input_path']
    input_shape = input_config['input_shape']
    one_img = wmli.imread(input_path)
    one_img = wmli.resize_and_pad(one_img, input_shape[2:][::-1],center_pad=False)
    if input_config.get("gray",False):
        one_img = wmli.nprgb_to_gray(one_img,keep_dim=True)
    one_img = one_img.transpose(2, 0, 1)
    if 'normalize_cfg' in input_config.keys() and input_config['normalize_cfg'] is not None:
        normalize_cfg = input_config['normalize_cfg']
        mean = np.array(normalize_cfg['mean'], dtype=np.float32)
        std = np.array(normalize_cfg['std'], dtype=np.float32)
        one_img = wtu.npnormalize(one_img,mean=mean,std=std)
    one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(False)
    return one_img