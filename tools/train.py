import argparse
import os
import os.path as osp
import time
import warnings
import torch
from thirdparty.pyconfig.config import Config 
import wtorch.train_toolkit as wtt
import wml_utils as wmlu
import wtorch.dist as wtd
import torch.distributed as dist
import torch.multiprocessing as mp
import socket


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        nargs="+",
        help="gpus for train")
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--use-fp16',
        action='store_true',
        help='Whether or not use fp16 for training')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Whether or not use enable debug')
    parser.add_argument('--dist-port', default="12355", help='port for disttribute training')
    parser.add_argument('--begin-iter', type=int, default=0)
    parser.add_argument('--workers-per-gpu', type=int, default=-1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main(rank,world_size,args):
    #wtd.configure_nccl()
    #wtd.configure_omp()
    #wtd.configure_module()
    from mmdet.utils import (get_device, get_root_logger,
                             replace_cfg_vals)
    from mmdet.utils.datadef import set_debug
    from mmdet.apis import init_random_seed, set_random_seed 
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.engine.train_loop import SimpleTrainer

    if world_size>1:
        wtd.setup_dist_group(rank,world_size,port=args.dist_port)
        torch.cuda.set_device(rank)
        device = rank
    else:
        device = torch.device('cuda')
    print(f"Host name {socket.gethostname()}")
    print(f"Config path {args.config}")
    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    set_debug(args.debug)

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or '
                          '"auto_scale_lr.enable" or '
                          '"auto_scale_lr.base_batch_size" in your'
                          ' configuration file. Please update all the '
                          'configuration files to mmdet >= 2.24.1.')

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
        print(f"Update work dir to {cfg.work_dir}")
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
        print(f"Set work dir to {cfg.work_dir}")
    
    if args.workers_per_gpu >= 0:
        cfg.data.workers_per_gpu = args.workers_per_gpu
        print(f"Update data workers_per_gpu to {cfg.data.workers_per_gpu}")
    
    if args.use_fp16:
        cfg.work_dir = cfg.work_dir+"_fp16"
        print(f"Update work dir to {cfg.work_dir}")

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # create work_dir
    os.makedirs(osp.abspath(cfg.work_dir),exist_ok=True)

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info

    cfg.device = get_device()
    # set random seeds
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = wmlu.base_name(args.config)

    model = build_detector(cfg.model)
    model.init_weights()
    model.to(device)

    dataset = build_dataset(cfg.data.train)

    model.CLASSES = dataset.CLASSES

    trainer = SimpleTrainer(cfg,model,dataset,rank,max_iters=cfg.max_iters,
                            use_fp16=args.use_fp16,
                            world_size=world_size,
                            begin_iter=args.begin_iter,
                            meta=meta)
    trainer.run()


if __name__ == '__main__':
    args = parse_args()
    world_size = len(args.gpus)
    gpus_str = ",".join([str(x) for x in args.gpus])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_str
    print(os.environ['CUDA_VISIBLE_DEVICES'])


    #torch.set_num_threads(1)
    if len(args.gpus)>1:
        mp.spawn(main,args=(world_size,args),nprocs=world_size,join=True)
    else:
        main(0,world_size,args)
'''
python tools/trainv2.py configs/work/gds1/faster_rcnn.py --no-validate --gpu-id 2
python tools/train.py configs/aiot_project/b11act/faster_rcnn.py --gpus 2 3
'''