# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import warnings
from functools import partial
import torch
from mmcv import Config, DictAction
import os
from mmdet.core.export import build_model_from_cfg, preprocess_example_input
import wtorch.train_toolkit as wtt

os.environ['CUDA_VISIBLE_DEVICES'] = "2"


def pytorch2traced(model,
                 input_img,
                 input_shape, #(B,C,H,W)
                 normalize_cfg,
                 show=False,
                 output_file='tmp.torch',
                 test_img=None,
                 skip_postprocess=False):

    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }
    # prepare input
    one_img, one_meta = preprocess_example_input(input_config)
    #one_img = one_img.repeat(3,1,1,1)
    img_list = one_img.cuda()
    model.cuda()
    wtt.freeze_model(model)

    if skip_postprocess:
        warnings.warn('Not all models support export onnx without post '
                      'process, especially two stage detectors!')
        model.forward = model.forward_dummy
        traced = torch.jit.trace(model,one_img)
        traced.save(output_file)

        print(f'Successfully exported traced model without '
              f'post process: {output_file}')
        return

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(
        model.forward,
        img_metas=None,
        return_loss=False)
    print(f"input shape {img_list.shape}")
    traced = torch.jit.trace(model,(img_list,))
    traced.save(output_file)
    model.forward = origin_forward

    print(f'Successfully exported traced model: {output_file}')

def parse_normalize_cfg(test_pipeline):
    transforms = None
    for pipeline in test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if _['type'] == 'Normalize']
    assert len(norm_config_li) == 1, '`norm_config` should only have one'
    norm_config = norm_config_li[0]
    return norm_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to traced')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show onnx graph and detection outputs')
    parser.add_argument('--output-file', type=str, default='tmp.traced')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--test-img', type=str, default=None, help='Images for test')
    parser.add_argument(
        '--dataset',
        type=str,
        default='coco',
        help='Dataset name. This argument is deprecated and will be removed \
        in future releases.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        #default=[800, 1216],
        help='input image size')
    parser.add_argument(
        '--mean',
        type=float,
        nargs='+',
        default=[123.675, 116.28, 103.53],
        help='mean value used for preprocess input data.This argument \
        is deprecated and will be removed in future releases.')
    parser.add_argument(
        '--std',
        type=float,
        nargs='+',
        default=[58.395, 57.12, 57.375],
        help='variance value used for preprocess input data. '
        'This argument is deprecated and will be removed in future releases.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--skip-postprocess',
        action='store_true',
        help='Whether to export model without post process. Experimental '
        'option. We do not guarantee the correctness of the exported '
        'model.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    warnings.warn('Arguments like `--mean`, `--std`, `--dataset` would be \
        parsed directly from config file and are deprecated and \
        will be removed in future releases.')


    try:
        from mmcv.onnx.symbolic import register_extra_symbolics
    except ModuleNotFoundError:
        raise NotImplementedError('please update mmcv to version>=v1.0.4')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.shape is None:
        img_scale = cfg.img_scale
        input_shape = (1, 3, img_scale[0], img_scale[1])
    elif len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    
    print(f"input shape {input_shape}")

    # build the model and load checkpoint
    model = build_model_from_cfg(args.config, args.checkpoint,
                                 args.cfg_options)

    if not args.input_img:
        args.input_img = osp.join(osp.dirname(__file__), '../../demo/demo.jpg')

    normalize_cfg = cfg.img_norm_cfg

    # convert model to onnx file
    pytorch2traced(
        model,
        args.input_img,
        input_shape,
        normalize_cfg,
        show=args.show,
        output_file=args.output_file,
        test_img=args.test_img,
        skip_postprocess=args.skip_postprocess)