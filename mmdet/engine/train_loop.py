import torch
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
import wml_utils as wmlu
from mmdet.utils.lr_scheduler_toolkit import *
import wtorch.train_toolkit as wtt
import wtorch.utils as wtu
from .build import *
import time
import os
import sys
import wtorch.summary as summary
import random
from wtorch.utils import unnormalize
import object_detection2.visualization as odv
import object_detection2.bboxes as odb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import wtorch.dist as wtd
from wtorch.dropblock import LinearScheduler as DBLinearScheduler
import numpy as np
from mmdet.dataloader.build import build_dataloader
from mmdet.utils.datadef import *
from .base_trainer import BaseTrainer
import traceback
from mmdet.datasets.pipelines import Compose
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks



class SimpleTrainer(BaseTrainer):
    MAX_ERROR_STEP_NR = 100
    def __init__(self,cfg,model,dataset,rank,max_iters,world_size,use_fp16=False,begin_iter=1,meta={'exp_name':"mmdet"}):
        super().__init__(cfg)
        self.error_step_nr = 0
        self.model = model
        self.dataset = dataset
        self.rank = rank
        self.work_dir = cfg.work_dir
        self.estimate_time_cost = wmlu.EstimateTimeCost(total_nr=1)
        self.data_loader = build_dataloader(cfg.data.dataloader,
                                            dataset,
                                            cfg=cfg.data,
                                            samples_per_gpu=self.cfg.data.samples_per_gpu,
                                            num_workers=self.cfg.data.workers_per_gpu,
                                            pin_memory=self.cfg.data.pin_memory,
                                            dist=world_size>1,
                                            persistent_workers=False,
                                            )
        self.data_loader_iter = iter(self.data_loader)
        self.data_processor = DATAPROCESSOR_REGISTRY.get(cfg.data.data_processor) 
        self.iter = begin_iter
        self.max_iters = max_iters
        self.world_size = world_size
        self.use_fp16 = use_fp16
        self.step_modules = []
        self.init_before_run()
        self.run_info = {}
        self.meta = meta
        pass

    def init_before_run(self):
        self.error_step_nr = 0
        self.total_norm = None
        cfg = self.cfg
        model = wtu.get_model(self.model)
        if hasattr(cfg,"finetune_model"):
            if cfg.finetune_model:
                wtt.finetune_model(model,
                  names2train=cfg.get("names_2train",None),
                  names_not2train=cfg.get("names_not2train",None))

        if dist.is_available() and self.world_size>1:
            model = wtd.convert_sync_batchnorm(model)
            pass
        bn_momentum = cfg.get("bn_momentum",None)
        if bn_momentum is not None:
            print(f"Set bn momentum to {bn_momentum}")
            wtt.set_bn_momentum(model,bn_momentum)

        is_freeze_bn = cfg.get("freeze_bn",None)
        if is_freeze_bn is not None and is_freeze_bn:
            print(f"freeze bn")
            wtt.freeze_bn(model)

        if is_debug():
            print("register_forward_hook")
            #wtt.register_forward_hook(model,wtt.isfinite_hook)
        
        self.max_norm = None
        max_norm = cfg.model.train_cfg.get("max_norm",16.0)
        if max_norm>0.1:
            print(f"Enable grad clip, Set max_norm to {max_norm}")
            self.max_norm = max_norm
        else:
            print(f"Disable grad clip")

        print("model parameters info")
        wtt.show_model_parameters_info(model)
        print(f"sync norm states")
        wtt.show_async_norm_states(model)

        self.load_checkpoint(model,cfg)

        self.optimizer,self.unbn_weights,self.unweights,self.unbiases = build_optimizer(self.cfg, model)
        lr_scheduler_args = self.cfg.lr_config
        policy_name = lr_scheduler_args.pop("policy")
        '''lr_scheduler_args = prepare_lr_scheduler_args(lr_scheduler_args,
                                                       begin_epoch=begin_epoch,
                                                       epochs_num=cfg.solver.epochs_num,
                                                       dataset_len=len(train_dataset),
                                                       batch_size=batch_size)'''
        lr_scheduler_args['optimizer'] = self.optimizer
        self.lr_scheduler = build_lr_scheduler(policy_name,lr_scheduler_args)
        print(f"PID {os.getpid()}")


        if self.world_size>1:
            print(f"Use DDP model.")
            model = DDP(model,device_ids=[self.rank],output_device=self.rank)
            if self.use_fp16:
                model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
        self._raw_model = wtu.get_model(model)
        self.model = model

        if self.use_fp16:
            print(f"Use fp16")
            self.scaler = torch.cuda.amp.GradScaler()

        self.call_hook('before_run')
        self.estimate_time_cost.reset(self.max_iters-self.iter)

        for m in self.model.modules():
            if isinstance(m,DBLinearScheduler):
                self.step_modules.append(m)
        batch_pipeline = self.cfg.get("batch_pipeline",None) 
        if batch_pipeline is not None:
            self.batch_pipeline = Compose(batch_pipeline)
        else:
            self.batch_pipeline = None

        self.filename_counter = wmlu.Counter()
        if self.rank != 0:
            return

        print("Model")
        print(model)

        logdir = osp.join(self.work_dir,"tblog")
        print(f"log dir {logdir}")
        wmlu.create_empty_dir_remove_if(logdir,"tblog")
        self.log_writer = SummaryWriter(logdir)

    def run(self):
        while self.iter <= self.max_iters:
            try:
                self.run_info = {}
    
                self.before_iter()
    
                time_b0 = time.time()
                self.fetch_iter_data()
                time_b1 = time.time()
                self.run_info['data_time'] = time_b1-time_b0

                if self.use_fp16:
                    self.train_amp_one_iter()
                else:
                    self.train_one_iter()

                time_b2 = time.time()
                self.estimate_time_cost.add_count()
                self.run_info['model_time'] = time_b2-time_b1
    
                self.after_iter()
    
                self.save_checkpoint()
    
                self.run_info['iter_time'] = time.time()-time_b0
    
                self.log_after_iter()
    
                self.iter += 1
                self.error_step_nr = max(self.error_step_nr-2,0)
            except Exception as e:
                print(f"Train error iter={self.iter}, error nr = {self.error_step_nr}, {e}")
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()
                torch.cuda.empty_cache()
                self.error_step_nr += 1
            except:
                print(f"Train error iter={self.iter}, error nr = {self.error_step_nr}.")
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()
                torch.cuda.empty_cache()
                self.error_step_nr += 1
            if self.error_step_nr > SimpleTrainer.MAX_ERROR_STEP_NR:
                print("ERROR: too many errors, stop training.")
                break

        self.save_checkpoint()
        self.after_run()

    
    def fetch_iter_data(self):
        if not self.data_loader._DataLoader__initialized:
            print(f"Reinit data loader iter")
            if self.data_loader_iter is not None:
                try:
                    self.data_loader_iter._shutdown_workers()
                except:
                    pass
                del self.data_loader_iter
            self.data_loader_iter = iter(self.data_loader)
        data_batch = next(self.data_loader_iter)
        if self.data_processor is not None:
            data_batch = self.data_processor(data_batch,self.rank,pipeline=self.batch_pipeline)

        #data_batch = wtu.to(data_batch,device=self.rank)
        #print(self.rank,wtu.simple_model_device(self.model),data_batch['img'].device)
        #if is_debug():
        #if True:
        if is_debug():
            ann_dir = self.cfg.get("data_root",None)
            for img_meta in data_batch['img_metas']:
                filename = img_meta['filename']
                if ";" in filename:
                    filename = filename.split(";")
                else:
                    filename = [filename]
                for x in filename:
                    key = wmlu.get_relative_path(x,ann_dir)
                    self.filename_counter.add(key)

            if self.iter%500== 0:
                wmlu.show_dict(self.filename_counter)
                
        self.inputs = data_batch
    

    def train_one_iter(self):
        data_batch = self.inputs
        outputs = self._raw_model.train_step(data_batch, self.optimizer)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        self.outputs = outputs
        loss = outputs["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_norm is not None:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm, norm_type=2)
        self.optimizer.step()
        self.lr_scheduler.step()

    def train_amp_one_iter(self):
        data_batch = self.inputs
        with torch.cuda.amp.autocast():
            outputs = self._raw_model.train_step(data_batch, self.optimizer)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        self.outputs = outputs
        loss = outputs["loss"]
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        if self.max_norm is not None:
            self.scaler.unscale_(self.optimizer)
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm, norm_type=2)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_scheduler.step()

    def log_after_iter(self):
        if self.iter%self.cfg.log_config.print_interval == 0:
            data_time = self.run_info['data_time']
            model_time = self.run_info['model_time']
            iter_time = self.run_info['iter_time']
            print(f"{self.meta['exp_name']}[{self.iter}/{self.max_iters}], loss={self.outputs['loss']:.3f}, data_time={data_time:.3f}, model_time={model_time:.3f}, iter time={iter_time: .3f}, {self.estimate_time_cost}")
            sys.stdout.flush()

        if self.iter%self.cfg.log_config.tb_interval == 0:
            self.tblog()
            torch.cuda.empty_cache()
    
    def tblog(self):
        try:
            self.__tblog()
        except Exception as e:
            print(f"TBLOG ERROR: {e}")
            
    def __tblog(self):
        if self.rank!=0:
            return
        global_step = self.iter
        for tag, val in self.outputs["log_vars"].items():
            if isinstance(val, str):
                self.log_writer.add_text(tag, val, global_step)
            else:
                self.log_writer.add_scalar(tag, val, global_step)

        self.tblog_imgs()

        for i,m in enumerate(self.step_modules):
            if isinstance(m,DBLinearScheduler):
                self.log_writer.add_scalar(f"drop_blocks/db{i}",m.dropblock.cur_drop_prob,global_step=global_step)

        if self.total_norm is not None:
            self.log_writer.add_scalar("total_norm",self.total_norm,global_step=global_step)
        model = wtu.get_model(self.model)
        summary.log_all_variable(self.log_writer,model,global_step=global_step)
        summary.log_optimizer(self.log_writer,self.optimizer,global_step,name=self.meta['exp_name'])

    def tblog_imgs(self):
        global_step = self.iter
        img_nr = self.cfg.log_config.get('img_nr',3)
        imgs = self.inputs['img']
        self.log_writer.add_histogram("img",imgs,global_step)
        log_imgs_ = self.cfg.log_config.get("log_imgs",False)
        log_gt_imgs_ = self.cfg.log_config.get("log_gt_imgs",True)

        if not log_imgs_ and not log_gt_imgs_:
            return

        if imgs.shape[1] == 1:
            imgs = torch.tile(imgs,[1,3,1,1])
        #print(imgs.shape)
        idxs = list(range(imgs.shape[0]))
        random.shuffle(idxs)
        idxs = idxs[:img_nr]

        if 'img_norm_cfg' in self.cfg:
            mean = self.cfg.img_norm_cfg.mean
            std = self.cfg.img_norm_cfg.std
            is_rgb = self.cfg.img_norm_cfg.to_rgb
        else:
            mean = None
            std = None
            is_rgb = None

        for idx in idxs:
            img = imgs[idx].cpu()
            if mean is not None:
                img = unnormalize(img,mean=mean,std=std).cpu().numpy()
                img = np.transpose(img,[1,2,0])
                if not is_rgb:
                    img = img[...,::-1]
            else:
                img = np.transpose(img,[1,2,0])
            img = np.ascontiguousarray(img)
            img = np.clip(img,0,255).astype(np.uint8)
            raw_img = img.copy()
            if log_imgs_:
                self.log_writer.add_image(f"input/img{idx}",raw_img,global_step=global_step,
                    dataformats="HWC")
            if not log_gt_imgs_:
                continue
            gt_bboxes = self.inputs['gt_bboxes']
            gt_labels = self.inputs['gt_labels']
            gt_masks = self.inputs.get('gt_masks',None)
            gt_bboxes = gt_bboxes[idx].cpu().numpy()
            gt_labels = gt_labels[idx].cpu().numpy()
            if gt_masks is not None:
                gt_masks = gt_masks[idx].masks
            gt_bboxes = odb.npchangexyorder(gt_bboxes)
            #debug
            #gt_labels = np.array(list(range(gt_labels.shape[0])))+gt_labels.shape[0]*10
            if gt_labels.shape[0]>0:
                img = odv.draw_text_on_image(img,gt_labels.shape[0])

            if gt_masks is not None:
                img = odv.draw_bboxes_and_maskv2(img,gt_labels,bboxes=gt_bboxes,masks=gt_masks,
                                                 is_relative_coordinate=False,show_text=True)
            else:
                img = odv.draw_bboxes(img,gt_labels,bboxes=gt_bboxes,is_relative_coordinate=False)
            self.log_writer.add_image(f"input/img_with_bboxes{idx}",img,global_step=global_step,
                    dataformats="HWC")

    
    def save_checkpoint(self):
        if not self.iter%self.cfg.checkpoint_config.interval==0 or self.rank!=0:
            return
        model = wtu.get_model(self.model)
        states = {
                'iter': self.iter,
                'state_dict': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        output_path = osp.join(self.cfg.work_dir,"weights",f"checkpoint_{self.iter}.pth")
        wmlu.make_dir_for_file(output_path)
        print(f"Save ckpt {output_path}")
        torch.save(states, output_path)
        sym_path = wmlu.change_name(output_path,basename="latest")
        wmlu.symlink(output_path,sym_path)


    def after_run(self):
        self.call_hook('after_run')

    def before_iter(self):
        self.total_norm = None
        self.call_hook('before_iter')
        for m in self.step_modules:
            m.step(self.iter)

    def after_iter(self):
        self.call_hook('after_iter')

    def load_checkpoint(self,model,cfg):
        if cfg.load_from is not None:
            if isinstance(cfg.load_from,str):
                self.load_checkpoint_from_one_file(model,cfg.load_from)
            elif isinstance(cfg.load_from,(list,tuple)):
                for path in cfg.load_from:
                    self.load_checkpoint_from_one_file(model,path)
            else:
                print(f"ERROR: error load from value {cfg.load_from}")

    def load_checkpoint_from_one_file(self,model,path):
        path = get_weight_path(path)
        if osp.exists(path):
            print(f"Load from {path}")
            data = torch.load(path)
            if "state_dict" in data:
                data = data["state_dict"]
            
            wtu.forgiving_state_restore(model,data) 
        else:
            print(f"ERROR: {path} not exists.")

