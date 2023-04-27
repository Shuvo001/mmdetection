# coding=utf-8
from argparse import ArgumentParser
from object_detection2.standard_names import *
import object_detection2.bboxes as odb
import os
import wml_utils as wmlu
import img_utils as wmli
import object_detection2.visualization as odv
import numpy as np
from iotoolkit.pascal_voc_toolkit import PascalVOCData,write_voc_xml
from iotoolkit.labelme_toolkit import LabelMeData,save_labelme_datav3
import os.path as osp
from itertools import count
import cv2
import shutil
import torch


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('data_dir', type=str,help='Path to test data dir')
    parser.add_argument('--checkpoint', default=None,type=str,help='Checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--img-suffix', type=str,default=".bmp;;.jpg;;.jpeg",help='img suffix')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    parser.add_argument('--gpus', default="0", type=str,help='Path to output file')
    parser.add_argument('--save-data-dir', type=str,help='Path to output file')
    parser.add_argument('--inplace', action='store_true',help='whether to save annotation inplace.')
    parser.add_argument('--save-scores', action='store_true',help='whether to save score.')
    parser.add_argument('--save-results',
        action='store_true',
        help='whether save results imgs.')
    args = parser.parse_args()
    return args

def save_annotation_masks(save_dir,img_path,img_shape,bboxes,labels,scores,det_masks,label_to_text):
    save_path = osp.join(save_dir,wmlu.base_name(img_path)+".json")
    image={'width':img_shape[1],'height':img_shape[0]}
    if osp.exists(save_path):
        nsave_path = wmlu.get_unused_path_with_suffix(save_path)
        print(f"ERROR: {save_path} exists, use new path {nsave_path}")
        save_path = nsave_path
    save_labelme_datav3(save_path,img_path,image,labels,bboxes,det_masks,label_to_text)

    return save_path

def save_annotation_bboxes(save_dir,img_path,img_shape,bboxes,labels,scores,det_masks,label_to_text,save_scores=False):
    save_path = osp.join(save_dir,wmlu.base_name(img_path)+".xml")
    if osp.exists(save_path):
        nsave_path = wmlu.get_unused_path_with_suffix(save_path)
        print(f"ERROR: {save_path} exists, use new path {nsave_path}")
        save_path = nsave_path
    labels_text = [label_to_text[x] for x in labels]
    bboxes = odb.npchangexyorder(bboxes)
    if save_scores:
        write_voc_xml(save_path,
                      img_path,img_shape,bboxes,labels_text,
                      probs=scores,
                      is_relative_coordinate=False)
    else:
        write_voc_xml(save_path,img_path,img_shape,bboxes,labels_text,is_relative_coordinate=False)

    return save_path

def save_annotation_bboxes_txt_head(save_dir):
    save_txt_path = osp.join(save_dir,"submission.csv")
    print(f"Save txt path {save_txt_path}")
    data = "image_id,bbox,category_id,confidence\n"

    with open(save_txt_path,"a") as f:
        f.write(data)

def save_annotation_bboxes_txt(save_dir,img_path,img_shape,bboxes,labels,scores,det_masks,label_to_text,save_scores=False):
    save_txt_path = osp.join(save_dir,"submission.csv")
    try:
        img_id = int(wmlu.base_name(img_path))
    except:
        print(f"Get img id for {img_path} faild.")
        img_id = -1
    bboxes = bboxes.astype(np.int32)

    with open(save_txt_path,"a") as f:
        for i,l in enumerate(labels):
            bbox = bboxes[i]
            data = f"{img_id},\"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]\",{l},{scores[i]}\n"
            f.write(data)

def save_annotation(save_dir,img_path,img_shape,bboxes,labels,scores,det_masks,classes,save_scores=False):
    label_to_text = dict(zip(range(len(classes)),classes))
    if det_masks is not None:
        return save_annotation_masks(save_dir,img_path,img_shape,bboxes,labels,scores,det_masks,label_to_text)
    else:
        '''save_annotation_bboxes_txt(save_dir,
                                      img_path,img_shape,bboxes,labels,scores,det_masks,label_to_text,
                                      save_scores=save_scores)'''
        return save_annotation_bboxes(save_dir,
                                      img_path,img_shape,bboxes,labels,scores,det_masks,label_to_text,
                                      save_scores=save_scores)

def text_fn(label,probability):
    return f"{label}:{probability:.2f}"

def get_results(result,score_thr=0.05):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    scores = bboxes[...,-1]
    bboxes = bboxes[...,:4]
    index = scores>score_thr
    bboxes = bboxes[index]
    scores = scores[index]
    labels = labels[index]

    return bboxes,labels,scores
    
def label_trans(labels):
    return np.array([x+1 for x in labels])

def main():
    args = parse_args()

    if args.gpus is not None and len(args.gpus)>0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print(os.environ['CUDA_VISIBLE_DEVICES'])

    import wtorch.utils as wtu
    from mmdet.datasets.pipelines import Compose
    from mmdet.apis import (ImageInferencePipeline,
                        init_detector,get_test_img_scale)

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

    save_path = args.save_data_dir
    if save_path is None:
        save_path = osp.join(work_dir,"tmp","auto_annotation")

    test_data_dir = args.data_dir 
    print(f"test_data_dir: {test_data_dir}")
    files = wmlu.get_files(test_data_dir,suffix=args.img_suffix)

    wmlu.create_empty_dir_remove_if(save_path,key_word="tmp")
    #metrics = ClassesWiseModelPerformace(num_classes=len(classes),classes_begin_value=0,model_type=PrecisionAndRecall)
    #metrics = ClassesWiseModelPerformace(num_classes=len(classes),classes_begin_value=0,model_type=Accuracy,
    #model_args={"threshold":0.3})
    input_size = get_test_img_scale(model.cfg)
    print(f"input size={input_size}")
    #save_size = (1024,640) 
    save_size = None

    pipeline = Compose(model.cfg.test_pipeline)
    detector = ImageInferencePipeline(pipeline=pipeline)

    #save_annotation_bboxes_txt_head(save_path)

    save_size = input_size

    for i,full_path in enumerate(files):
        try:
            print(f"process {i}/{len(files)}")
            #
            #if 1 not in gt_labels:
                #continue
            '''
            contours, hierarchy = cv2.findContours(binary_masks[0], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            print(contours)
            tmp_img = wmli.imread(full_path)
            cv2.drawContours(tmp_img,contours,-1,(255,0,0),2)
            wmli.imwrite("a.jpg",tmp_img)'''
            #
            #if wmlu.base_name(full_path) != "B61C1Y0521B5BAQ03-aa-02_ALL_CAM00":
                #continue
            #if osp.basename(full_path) not in imgs17:
                #continue
            img = wmli.imread(full_path)
            bboxes,labels,scores,det_masks,result = detector(model,
                                                             img,
                                                             input_size=input_size,score_thr=args.score_thr)
            name = wmlu.base_name(full_path)
            if args.inplace:
                ann_save_dir = osp.dirname(full_path)
            else:
                ann_save_dir = save_path
    
            ann_path = save_annotation(ann_save_dir,
                                       full_path,img.shape,bboxes,labels,scores,det_masks,classes,
                                       save_scores=args.save_scores)
    
            if not args.inplace and len(labels)>0:
                suffix = osp.splitext(full_path)[1][1:]
                save_img_path = wmlu.change_suffix(ann_path,suffix)
                wmlu.try_link(full_path,save_img_path)
    
            if args.save_results:
                img_save_path = os.path.join(save_path,name+".jpg")
    
                if save_size is not None:
                    wmli.imwrite(img_save_path,wmli.imread(full_path),save_size)
                else:
                    shutil.copy(full_path,img_save_path)
        
                img_save_path = os.path.join(save_path,name+"_pred.jpg")
                img = wmli.imread(full_path)
                if save_size is not None:
                    img,r = wmli.resize_imgv2(img,save_size,return_scale=True)
                    t_bboxes = bboxes*r
                else:
                    t_bboxes = bboxes
                img = odv.draw_bboxes_xy(img,
                                         scores=scores,classes=labels,bboxes=t_bboxes,text_fn=text_fn,
                                         show_text=True)
                if det_masks is not None:
                    img = odv.draw_mask_xy(img,classes=labels,bboxes=t_bboxes,masks=det_masks,color_fn=odv.red_color_fn)
                wmli.imwrite(img_save_path,img)
        except:
            pass


    
    print(f"Image save path: {save_path}, total process {len(files)}")
        
    print(classes)

if __name__ == "__main__":
    main()

'''
python tools/auto_annotation.py configs/aiot_project/b7mura/rcnn_yoloxv2_huges_redbc.py /home/wj/ai/mldata1/B7mura/datas/verify_0328 --score-thr 0.4 --work-dir 1 --gpus 0 --save-scores
'''
