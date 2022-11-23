def prepare_lr_scheduler_args(args,begin_epoch,epochs_num,dataset_len,batch_size):
    #epochs_num-begin_epoch-1)
    total_iters = int((epochs_num-begin_epoch)*dataset_len/batch_size)
    one_epoch_iters = int(dataset_len/batch_size)
    res = {}
    for k,v in args.items():
        if not isinstance(v,str):
            res[k] = v
            continue
        v = v.format(total_iters=total_iters,
                     begin_epoch=begin_epoch,
                     epochs_num=epochs_num,
                     one_epoch_iters=one_epoch_iters)
        v = float(eval(v))
        res[k] = v
    
    return res