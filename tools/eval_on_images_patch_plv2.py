# coding=utf-8
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from mmdet.apis.img_patch_inference import *
from object_detection2.standard_names import *
import object_detection2.bboxes as odb
import os
import wml_utils as wmlu
import img_utils as wmli
import object_detection2.visualization as odv
import numpy as np
from iotoolkit.coco_toolkit import COCOData
from iotoolkit.pascal_voc_toolkit import PascalVOCData
from iotoolkit.labelme_toolkit import LabelMeData
from object_detection2.metrics.toolkit import *
import os.path as osp
from itertools import count
import shutil
import pickle
from mmdet.datasets.pipelines import Compose

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('--checkpoint', default=None,help='Checkpoint file')
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
    parser.add_argument('--save_data_dir', type=str,help='Path to output file')
    parser.add_argument('--test_data_dir', type=str,help='Path to output file')
    parser.add_argument('--save_results',
        action='store_true',
        help='whether save results imgs.')
    args = parser.parse_args()
    return args

def eval_dataset(data_dir,classes):
    '''data = COCOData()
    data.read_data(wmlu.home_dir("ai/mldata/coco/annotations/instances_val2014.json"),
                   image_dir=wmlu.home_dir("ai/mldata/coco/val2014"))'''
    if classes is not None:
        text2label = dict(zip(classes,count()))
    else:
        text2label = {"scratch":0}

    print(f"Text to label")
    wmlu.show_dict(text2label)


    #data = PascalVOCData(label_text2id=label_text2id,absolute_coord=True)
    #data.read_data(data_dir,img_suffix=".bmp;;.jpg;;.jpeg",check_xml_file=False)
    data = LabelMeData(label_text2id=text2label,
                       absolute_coord=True,filter_empty_files=True)
    data.read_data(data_dir,img_suffix=".bmp;;.jpg;;.jpeg")

    return data

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

imgs17 = ["B68G1X0012C3AAN05-02_ALL_CAM00.bmp",
"B68G1X0012C3AAM05-02_ALL_CAM00.bmp",
"B68G190084B5BAC03-02_ALL_CAM00.bmp",
"B68G1X0012B6AAC02-02_ALL_CAM00.bmp",
"B68G1X0010B5BAK04-02_ALL_CAM00.bmp",
"B68G1X0004B7AAE01-02_ALL_CAM00.bmp",
"B68G190100B7AAK04-02_ALL_CAM00.bmp",
"B68G190100BXAAS02-02_ALL_CAM00.bmp",
"B68G190089A6BAP01-02_ALL_CAM00.bmp",
"B68G1X0029C6BAK02-02_ALL_CAM00.bmp",
"B68G190084C1BAF03-02_ALL_CAM00.bmp",
"B68G190084C4BAE04-02_ALL_CAM00.bmp",
"B68G190084B8BAP03-02_ALL_CAM00.bmp",
"B68G190084B7BAG02-02_ALL_CAM00.bmp",
"B68G190084B7BAC05-02_ALL_CAM00.bmp",
"B68G190084B7BAH04-02_ALL_CAM00.bmp",
"B68G190084B8BAG05-02_ALL_CAM00.bmp",
]
#imgs17 = ["B68G1X0004B7AAE01-02_ALL_CAM00.bmp"]

def main():
    args = parse_args()

    if args.gpus is not None and len(args.gpus)>0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, None, device=args.device)
    if args.checkpoint is None:
        checkpoint = osp.join(model.cfg.work_dir,"weights","latest.pth")
        if not osp.exists(checkpoint):
            checkpoint = osp.join(model.cfg.work_dir+"_fp16","weights","latest.pth")
    else:
        checkpoint = args.checkpoint

    print(f"Load {checkpoint}")
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

    work_dir = model.cfg.work_dir
    save_path = args.save_data_dir
    if save_path is None:
        save_path = osp.join(work_dir,"tmp","eval_on_images")

    test_data_dir = args.test_data_dir

    if test_data_dir is None:
        test_data_dir = model.cfg.data.val.data_dirs
    
    print(f"test_data_dir: {test_data_dir}")

    wmlu.create_empty_dir_remove_if(save_path,key_word="tmp")
    #metrics = COCOEvaluation(num_classes=len(classes),label_trans=label_trans)
    metrics = ClassesWiseModelPerformace(num_classes=len(classes),classes_begin_value=0,model_type=PrecisionAndRecall)
    #metrics = ClassesWiseModelPerformace(num_classes=len(classes),classes_begin_value=0,model_type=Accuracy,
                                          #model_args={"threshold":0.3})
    dataset = eval_dataset(test_data_dir,classes=classes)
    input_size = tuple(list(model.cfg.img_scale)[::-1]) #(h,w)->(w,h)
    #save_size = (1024,640) 
    save_size = None

    pipeline = Compose(model.cfg.test_pipeline)
    detector = ImagePatchInferencePipelineV2(patch_size=input_size[::-1],
                                          pipeline=pipeline)

    pyresults = []

    for i,data in enumerate(dataset.get_items()):
        full_path, shape, gt_labels, category_names, gt_boxes, binary_masks, area, is_crowd, num_annotations_skipped = data
        print(f"process {osp.basename(full_path)} {i}/{len(dataset)}")
        #if wmlu.base_name(full_path) != "B68G1X0029C6BAK02-02_ALL_CAM00":
            #continue
        #if 1 not in gt_labels:
            #continue
        #if osp.basename(full_path) not in imgs17:
            #continue
        gt_boxes = odb.npchangexyorder(gt_boxes)
        bboxes,labels,scores,det_masks,result = detector(model, full_path,input_size=None,
                                                         score_thr=args.score_thr)
        #if 1 not in labels:
            #continue
        name = wmlu.base_name(full_path)
        if args.save_results:
            img_save_path = os.path.join(save_path,name+".jpg")

            if save_size is not None:
                wmli.imwrite(img_save_path,wmli.imread(full_path),save_size)
            else:
                wmli.read_and_write_img(full_path,img_save_path)
    
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

            img_save_path = os.path.join(save_path,name+"_gt.jpg")
            img = wmli.imread(full_path)
            if save_size is not None:
                img,r = wmli.resize_imgv2(img,save_size,return_scale=True)
                t_gt_boxes = gt_boxes*r
            else:
                t_gt_boxes = gt_boxes
            img = odv.draw_bboxes_xy(img,classes=gt_labels,bboxes=t_gt_boxes,text_fn=text_fn)
            wmli.imwrite(img_save_path,img)

        kwargs = {}
        kwargs['gtboxes'] = gt_boxes
        kwargs['gtlabels'] =gt_labels 
        kwargs['boxes'] = bboxes
        kwargs['labels'] =  labels
        kwargs['probability'] =  scores
        kwargs['img_size'] = shape
        kwargs['use_relative_coord'] = False

        pyresults.append(copy.deepcopy(kwargs))
        metrics(**kwargs)
        
        if i%100 == 99:
            metrics.show()
    
    print(f"Image save path: {save_path}, total process {len(dataset)}")

    with open(osp.join(save_path,"results.pk"),"wb") as f:
        pickle.dump(pyresults,f)
        
    metrics.show()
    print(classes)

if __name__ == "__main__":
    main()

'''
python tools/eval_on_images.py configs/work/gds1/faster_rcnn.py /home/wj/ai/mldata1/GDS1Crack/mmdet/weights/latest.pth --test_data_dir /home/wj/ai/mldata1/GDS1Crack/val/ng --gpus 3
||0.128|0.231|
python tools/eval_on_images.py configs/aiot_project/b11act/faster_rcnn.py ~/ai/mldata1/B11ACT/workdir/b11act/weights/latest.pth --gpus 1  
||0.817|1.000|
python tools/eval_on_images.py configs/aiot_project/b11act/faster_rcnn.py ~/ai/mldata1/B11ACT/workdir/b11act_new/weights/checkpoint_49000.pth 
|0.626|1.000|
python tools/eval_on_images.py configs/aiot_project/b11act/faster_rcnn.py ~/ai/mldata1/B11ACT/workdir/b11act_new/weights/checkpoint_49000.pth 
||0.619|1.000|
python tools/eval_on_images.py configs/aiot_project/b11act/faster_rcnn.py /home/wj/ai/mldata1/B11ACT/workdir/b11act_new_fp16/weights/checkpoint_49000.pth
||0.626|1.000|
'''
