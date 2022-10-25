# coding=utf-8
# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
from mmdet.apis import (async_inference_detector, inference_detector,inference_detectorv2,
                        init_detector, show_result_pyplot)
import torch
import wtorch.utils as wtu
import object_detection2.bboxes as odb
import os
import wml_utils as wmlu
import img_utils as wmli
import object_detection2.visualization as odv
import numpy as np
from iotoolkit.coco_toolkit import COCOData
from iotoolkit.pascal_voc_toolkit import PascalVOCData
from object_detection2.metrics.toolkit import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('model', help='traced model file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    parser.add_argument('--gpus', default="1", type=str,help='Path to output file')
    parser.add_argument('--save_data_dir', default="/home/wj/ai/mldata1/GDS1Crack/tmp/predict_on_images", type=str,help='Path to output file')
    parser.add_argument('--test_data_dir', default="demo", type=str,help='Path to output file')
    args = parser.parse_args()
    return args


def text_fn(label,probability):
    return f"{label}:{probability:.2f}"

def inference_traced(model, img,mean=None,std=None,input_size=(1024,1024),score_thr=0.05):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    device = torch.device("cuda:0")

    if not isinstance(img, np.ndarray):
        filename = img
        img = wmli.imread(img)
    
    img,r = wmli.resize_imgv2(img,input_size,return_scale=True)
    img = torch.tensor(img,dtype=torch.float32)
    img = img.permute(2,0,1)
    img = torch.unsqueeze(img,dim=0)
    img = img.to(device)

    if mean is not None:
        img = wtu.normalize(img,mean,std)
    
    img = wtu.pad_feature(img,input_size,pad_value=0)

    with torch.no_grad():
        results = model(img)

    results = results.cpu().numpy()
    mask = results[...,4]>score_thr
    results = results[mask]
    bboxes = results[...,:4]
    scores = results[...,4]
    labels = results[...,5].astype(np.int32)
    bboxes = bboxes/r

    return bboxes,labels,scores,results

def main():
    args = parse_args()

    if args.gpus is not None and len(args.gpus)>0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # build the model from a config file and a checkpoint file
    print(f"Load {args.model}")
    model = torch.jit.load(args.model,map_location=args.device)
    files = wmlu.recurse_get_filepath_in_dir(args.test_data_dir,suffix=".bmp;;.jpg")

    save_path = args.save_data_dir

    wmlu.create_empty_dir_remove_if(save_path,key_word="tmp")
    print(f"Inference {args.test_data_dir}")

    mean=[123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375]
    input_size = (1024,1024)
    input_size = (1216,800)
    
    for i,full_path in enumerate(files):
        bboxes,labels,scores,result = inference_traced(model, full_path,mean=mean,std=std,input_size=input_size,score_thr=args.score_thr)
        name = wmlu.base_name(full_path)
        img_save_path = os.path.join(save_path,name+".png")
        bboxes = odb.npchangexyorder(bboxes)
        img = wmli.imread(full_path)
        r_img = odv.draw_bboxes(img,labels,scores,bboxes,
                                 text_fn=text_fn,
                                 show_text=True,is_relative_coordinate=False,
                                 thickness=2,font_scale=0.6)
        print(f"Save {img_save_path}")
        wmli.imwrite(img_save_path,r_img)

if __name__ == "__main__":
    main()

'''
python tools/predict_on_images.py configs/work/gds1/faster_rcnn.py /home/wj/ai/mldata1/GDS1Crack/mmdet/weights/latest.pth --test_data_dir /home/wj/ai/mldata1/GDS1Crack/val/ng --gpus 2
'''
