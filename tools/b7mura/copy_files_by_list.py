import wml_utils as wmlu
import argparse
import random
import os.path as osp
import os
import shutil
from iotoolkit.pascal_voc_toolkit import write_voc_xml
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--dir_path', default="/home/wj/ai/mldata1/B7mura/feedback/0512_list",help='train config file path')
    parser.add_argument('--img_path', default="/opt/DataDisk/zhangym/tsp_ftp/imgs",help='train config file path')
    parser.add_argument('--save_path', default="/home/wj/ai/mldata1/B7mura/feedback",help='train config file path')
    parser.add_argument('--sample_nr', type=int,default=2000,help='train config file path')
    return parser.parse_args()

def read_list_txt(path):
    res = []
    with open(path,"r") as f:
        data = f.readlines()
        for line in data:
            idx = line.rfind(";")
            if idx<0:
                continue
            base_info = line[:idx].split(";")[1]
            base_info = base_info.split("/")[3]
            line = line[idx+1:]
            line = line.split("\t")
            filename = line[0]
            if filename[-4:]  == ".JPG":
                continue
            filename = filename.replace("MURA",base_info+"_SCANIMAGE")
            type = line[1]
            if "MU4U" in type:
                res.append(filename)
    
    return res

def read_list_files(dir):
    files = wmlu.get_files(dir,suffix=".txt")
    res = []
    for f in files:
        res.extend(read_list_txt(f))
    return res

def get_imgs(dir):
    files = wmlu.get_files(dir,suffix=".jpg")
    res = {}
    for f in files:
        res[osp.basename(f)] = f
    
    return res

if __name__ == "__main__":
    args = parse_args()
    data = read_list_files(args.dir_path)
    random.shuffle(data)
    data = data[:args.sample_nr]
    imgs = get_imgs(args.img_path)
    save_path = args.save_path
    save_path_name = wmlu.base_name(args.dir_path)
    save_path = osp.join(save_path,save_path_name+"_imgs")
    os.makedirs(save_path,exist_ok=True)
    bboxes = np.zeros([0,4])
    labels_text = []

    for fn in data:
        if fn not in imgs:
            print(f"ERROR: Find {fn} in imgs faild.")
            continue
        shutil.copy(imgs[fn],save_path)
        save_file_path = osp.join(save_path,osp.basename(imgs[fn]))
        xml_save_path = wmlu.change_suffix(save_file_path,"xml")
        write_voc_xml(xml_save_path,save_file_path,[0,0], bboxes, labels_text)


