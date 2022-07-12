# coding=utf-8
# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
from mmdet.apis import (async_inference_detector, inference_detector,inference_detectorv2,
                        init_detector, show_result_pyplot)
import object_detection2.config.config as config
from object_detection2.standard_names import *
from object_detection2.engine.defaults import default_argument_parser, get_config_file
from object_detection2.data.dataloader import *
from object_detection2.data.datasets.build import DATASETS_REGISTRY
import object_detection2.bboxes as odb
import tensorflow as tf
import os
from object_detection_tools.predictmodel import PredictModel
import wml_utils as wmlu
import img_utils as wmli
import object_detection2.visualization as odv
import numpy as np
from object_detection2.data.datasets.buildin import coco_category_index
from iotoolkit.coco_toolkit import COCOData
from iotoolkit.pascal_voc_toolkit import PascalVOCData
from object_detection2.metrics.toolkit import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.05, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    parser.add_argument('--gpus', default="1", type=str,help='Path to output file')
    parser.add_argument('--save_data_dir', default="/home/wj/ai/mldata1/GDS1Crack/tmp/eval_on_images", type=str,help='Path to output file')
    parser.add_argument('--test_data_dir', default="demo", type=str,help='Path to output file')
    args = parser.parse_args()
    return args


def text_fn(label,probability):
    return f"{label}:{probability:.2f}"

def main(_):
    args = parse_args()

    if args.gpus is not None and len(args.gpus)>0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    files = wmlu.recurse_get_filepath_in_dir(args.test_data_dir,suffix=".bmp;;.jpg")
    # test a single image

    save_path = args.save_data_dir

    wmlu.create_empty_dir_remove_if(save_path,key_word="tmp")

    mean=[123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375]
    input_size = (1024,1024)
    input_size = (1216,800)
    
    for i,full_path in enumerate(files):
        bboxes,labels,scores,result = inference_detectorv2(model, full_path,mean=mean,std=std,input_size=input_size,score_thr=args.score_thr)
        name = wmlu.base_name(full_path)
        img_save_path = os.path.join(save_path,name+".png")
        bboxes = odb.npchangexyorder(bboxes)
        img = wmli.imread(full_path)
        r_img = odv.draw_bboxes(img,labels,scores,bboxes,
                                             text_fn=text_fn,
                                             show_text=True,is_relative_coordinate=False)
        wmli.imwrite(img_save_path,r_img)

if __name__ == "__main__":
    tf.app.run()

'''
python object_detection_tools/eval_on_images.py --test_data_dir ~/ai/mldata1/GDS1Crack/val/ng/ --gpus 3 --config-file gds1 --save_data_dir ~/ai/mldata1/GDS1Crack/tmp/gds1_output
0.114|0.171 60000
0.111|0.164 100000
0.116|0.175 50000
python object_detection_tools/eval_on_images.py --test_data_dir ~/ai/mldata1/GDS1Crack/val/ng/ --gpus 3 --config-file gds1v2 --save_data_dir ~/ai/mldata1/GDS1Crack/tmp/gds1_output
0.141|0.213 24999
||0.145|0.222| 49999
'''
'''
python -m tf2onnx.convert --graphdef tensorflow-model-graphdef-file --output model.onnx --inputs input0:0,input1:0 --outputs output0:0
'''
