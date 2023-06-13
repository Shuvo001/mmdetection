# coding=utf-8
from argparse import ArgumentParser
from object_detection2.standard_names import *
import object_detection2.bboxes as odb
import sys
import os
import wml_utils as wmlu
import img_utils as wmli
import object_detection2.visualization as odv
import numpy as np
from iotoolkit.pascal_voc_toolkit import read_voc_xml
import os.path as osp
from itertools import count
import cv2
import random
import shutil
import torch


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('data_dir', type=str,help='Path to test data dir')
    parser.add_argument('--checkpoint', default=None,type=str,help='Checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--img-suffix', type=str,default=".bmp;;.jpg;;.jpeg;;.png",help='img suffix')
    parser.add_argument(
        '--score-thr', type=float, default=0.4, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    parser.add_argument('--gpus', default="0", type=str,help='Path to output file')
    parser.add_argument('--save-data-dir', type=str,help='Path to output file')
    parser.add_argument('--test-nr', type=int,help='Path to output file')
    parser.add_argument('--inplace', action='store_true',help='whether to save annotation inplace.')
    parser.add_argument('--shuffle', action='store_true',help='whether to shuffle input files.')
    parser.add_argument('--label', type=str,default="MP1U",help='label to patch.')
    parser.add_argument('--save-scores', action='store_true',help='whether to save score.')
    parser.add_argument('--save-results',
        action='store_true',
        help='whether save results imgs.')
    parser.add_argument('--copy-imgs',
        action='store_true',
        help='whether save copy imgs.')
    args = parser.parse_args()
    return args

def save_img_patch(img_path,save_path,bbox,bbox_scale=1.1,min_bbox_size=128,label="MP1U"):
    bboxes = np.array([bbox])
    bboxes = odb.npscale_bboxes(bboxes,bbox_scale)
    if min_bbox_size>1:
        bboxes = odb.clamp_bboxes(bboxes,min_size=min_bbox_size)
    bboxes = bboxes.astype(np.int32)
    img = wmli.imread(img_path)
    base_name = wmlu.base_name(img_path)
    bbox = bbox.astype(np.int32)
    coord = [str(x) for x in bbox]
    coord = ",".join(coord)
    t_save_path = osp.join(save_path,f"{base_name}_bbox{coord}_{label}.jpg")
    simg = wmli.crop_img_absolute_xy(img,bboxes[0])
    wmli.imwrite(t_save_path,simg)

class LResize:
    def __init__(self,size) -> None:
        self.size = size #w,h
    
    def __call__(self,img):
        img = wmli.resize_img(img,self.size,keep_aspect_ratio=True)
        return wmli.nprgb_to_gray(img)

def main():
    args = parse_args()

    if args.gpus is not None and len(args.gpus)>0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print(os.environ['CUDA_VISIBLE_DEVICES'])

    import wtorch.utils as wtu
    from mmdet.datasets.pipelines import Compose
    from mmdet.apis import (ImageInferencePipeline,
                        init_detector,get_test_img_scale)
    from iotoolkit.imgs_reader_mt import ImgsReader

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, None, device="cuda:0")

    if args.work_dir is not None:
        if len(args.work_dir)==1 and not osp.isdir(args.work_dir):
            work_dir = model.cfg.work_dir+args.work_dir
        else:
            work_dir = args.work_dir
        print(f"Update work dir to {work_dir}")
    else:
        work_dir = model.cfg.work_dir

    if args.checkpoint is None:
        checkpoint = osp.join(work_dir+"_fp16","weights","latest.pth")
        if not osp.exists(checkpoint):
            checkpoint = osp.join(work_dir,"weights","latest.pth")
    else:
        checkpoint = args.checkpoint

    print(f"Load {checkpoint}")
    os.system(f"ls -l {checkpoint}")
    checkpoint = torch.load(checkpoint,map_location="cpu")
    wtu.forgiving_state_restore(model,checkpoint)
    '''
    nms_pre: 为每一层nms之前的最大值
    nms: 仅在每一层内部做
    min_bbox_size: 为宽高的最小值
    max_per_img: 上述处理后排序，最很分最高的max_per_img候选
    '''
    print("RPN Head test config")
    #rpn_head.test_cfg 配置时通过config.model.test_cfg.rpn配置
    wmlu.show_dict(model.rpn_head.test_cfg)
    '''
    score_thr: 仅当score大于score_thr才会留下
    nms: iou_threshold, nms iou阀值
    max_per_img: nms后只留下最多max_per_img个目标
    mask_thr_binary: mask threshold
    '''
    print("RCNN test config")
    #roi_head.test_cfg 配置时通过config.model.test_cfg.rcnn配置
    wmlu.show_dict(model.roi_head.test_cfg)

    if hasattr(model.cfg,"classes"):
        classes = model.cfg.classes
    text2id = dict(zip(classes,count()))
    if hasattr(model.cfg,"label_text2id"):
        text2id.update(model.cfg.label_text2id)
    print(f"text2id")
    wmlu.show_dict(text2id)
    mp1u_label = args.label
    mp1u_id = text2id[mp1u_label]
    print(f"MP1U ID: {mp1u_id}")


    test_data_dir = args.data_dir 
    print(f"test_data_dir: {test_data_dir}")
    #files = wmlu.get_files(test_data_dir,suffix=args.img_suffix)
    resizer = LResize(size=list(model.cfg.img_scale)[::-1])
    if args.test_nr is not None and args.test_nr>0:
        files = wmlu.get_files(test_data_dir)
        if args.shuffle:
            random.shuffle(files)
            print(f"Shuffle files")
        files = files[:args.test_nr]
        args.copy_imgs = True
        print(f"test nr is {args.test_nr}, files len is {len(files)}")
        reader = ImgsReader(files,shuffle=False,transform=resizer)
        sys.stdout.flush()
    else:
        reader = ImgsReader(test_data_dir,transform=resizer)

    save_path = args.save_data_dir
    if save_path is None:
        save_path = osp.join("/home/wj/ai/mldata1/B7mura","tmp","add_missed_mp1u_"+wmlu.base_name(test_data_dir))

    wmlu.create_empty_dir_remove_if(save_path,key_word="tmp")
    #metrics = ClassesWiseModelPerformace(num_classes=len(classes),classes_begin_value=0,model_type=PrecisionAndRecall)
    #metrics = ClassesWiseModelPerformace(num_classes=len(classes),classes_begin_value=0,model_type=Accuracy,
    #model_args={"threshold":0.3})
    input_size = get_test_img_scale(model.cfg)
    print(f"input size={input_size}")
    print(f"Save path {save_path}")
    sys.stdout.flush()

    pipeline = Compose(model.cfg.test_pipeline)
    detector = ImageInferencePipeline(pipeline=pipeline)

    sys.stdout.flush()
    min_test_bbox_size = 64
    min_save_img_size = 128
    bbox_scale = 1.1
    iou_threshold = 0.3

    for i,(full_path,img) in enumerate(reader):
        try:
            p_shape = img.shape
            org_shape = wmli.get_img_size(full_path)
            i_scale = org_shape[0]/p_shape[0]
            print(f"process {i}/{len(reader.dataset)}")
            if len(img) == 0:
                print(f"Read {full_path} faild.")
                continue
            if i%10 == 0:
                sys.stdout.flush()
            bboxes,labels,scores,det_masks,result = detector(model,
                                                             img,
                                                             input_size=input_size,score_thr=args.score_thr)
            if mp1u_id not in labels:
                continue
            bboxes = bboxes*i_scale
            mask = labels==mp1u_id
            if not np.any(mask):
                continue
            bboxes = bboxes[mask]
            clamped_bbox = odb.clamp_bboxes(bboxes,min_test_bbox_size)
            xml_path = wmlu.change_suffix(full_path,"xml")
            _, gtbboxes, gtlabels_text, difficult, truncated, _ = read_voc_xml(xml_path,absolute_coord=True)
            gtbboxes = odb.npchangexyorder(gtbboxes)
            gtbboxes = odb.clamp_bboxes(gtbboxes,min_test_bbox_size)

            iou_matrix = odb.iou_matrix(clamped_bbox,gtbboxes)
            added_bboxes = []
            for i in range(len(clamped_bbox)):
                cious = iou_matrix[i]
                if np.any(cious>iou_threshold):
                    continue
                bbox = bboxes[i]
                cbbox = clamped_bbox[i]
                if len(added_bboxes)>0:
                    sious = odb.iou_matrix([cbbox],added_bboxes)
                    if np.any(sious>iou_threshold):
                        continue
                added_bboxes.append(cbbox)
                save_img_patch(full_path,save_path,bbox,bbox_scale,min_save_img_size,label=mp1u_label)

        except Exception as e:
            print(e)
            pass

    
    print(f"Image save path: {save_path}, total process {len(reader.dataset)}")
        
    print(classes)

if __name__ == "__main__":
    main()

'''
python tools/auto_annotation.py configs/aiot_project/b7mura/rcnn_yoloxv2_huges_redbc.py /home/wj/ai/mldata1/B7mura/datas/verify_0328 --score-thr 0.4 --work-dir 1 --gpus 0 --save-scores
'''
