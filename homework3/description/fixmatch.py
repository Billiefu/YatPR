# import needed library
import os
import random

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from homework4.source.utils import net_builder, get_logger, count_parameters, over_write_args_from_file
from train_utils import TBLog, get_optimizer, get_cosine_schedule_with_warmup
from models.fixmatch.fixmatch import FixMatch
from datasets.ssl_dataset import SSL_Dataset, ImageNetLoader
from datasets.data_utils import get_data_loader


def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and args.resume == False:
        import shutil
        shutil.rmtree(save_path)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')

    # if args.seed is not None:
    #     warnings.warn('You have chosen to seed training. '
    #                   'This will turn on the CUDNN deterministic setting, '
    #                   'which can slow down your training considerably! '
    #                   'You may see unexpected behavior when restarting '
    #                   'from checkpoints.')

    # if args.gpu is not None:
    #     warnings.warn('You have chosen a specific GPU. This will completely '
    #                   'disable data parallelism.')

    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])

    # # distributed: true if manually selected or if world_size > 1
    # args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    # ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    # if args.multiprocessing_distributed:
    #     # now, args.world_size means num of total processes in all nodes
    #     args.world_size = ngpus_per_node * args.world_size

    #     # args=(,) means the arguments of main_worker
    #     mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # else:
    #     main_worker(args.gpu, ngpus_per_node, args)
    
    main_worker(args)


def main_worker(args):
    '''
    main_worker is conducted on each GPU.
    '''

    global best_acc1

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    # assert args.seed is not None
    args.seed=0
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    # # SET UP FOR DISTRIBUTED TRAINING
    # if args.distributed:
    #     if args.dist_url == "env://" and args.rank == -1:
    #         args.rank = int(os.environ["RANK"])
    #     if args.multiprocessing_distributed:
    #         args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

    #     # set distributed group:
    #     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                             world_size=args.world_size, rank=args.rank)

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    # if args.rank % ngpus_per_node == 0:
    #     tb_log = TBLog(save_path, 'tensorboard', use_tensorboard=args.use_tensorboard)
    #     logger_level = "INFO"
    tb_log = TBLog(save_path, 'tensorboard', use_tensorboard=args.use_tensorboard)
    logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    # logger.warning(f"USE GPU: {args.gpu} for training")

    # SET FixMatch: class FixMatch in models.fixmatch
    args.bn_momentum = 1.0 - 0.999
    if 'imagenet' in args.dataset.lower():
        _net_builder = net_builder('ResNet50', False, None, is_remix=False)
    else:
        _net_builder = net_builder(args.net,
                                   args.net_from_name,
                                   {'first_stride': 2 if 'stl' in args.dataset else 1,
                                    'depth': args.depth,
                                    'widen_factor': args.widen_factor,
                                    'leaky_slope': args.leaky_slope,
                                    'bn_momentum': args.bn_momentum,
                                    'dropRate': args.dropout,
                                    'use_embed': False,
                                    'is_remix': False},
                                   )

    model = FixMatch(_net_builder,
                     args.num_classes,
                     args.ema_m,
                     args.T,
                     args.p_cutoff,
                     args.ulb_loss_ratio,
                     args.hard_label,
                     num_eval_iter=args.num_eval_iter,
                     tb_log=tb_log,
                     logger=logger)

    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')

    # SET Optimizer & LR Scheduler
    ## construct SGD and cosine lr scheduler
    optimizer = get_optimizer(model.model, args.optim, args.lr, args.momentum, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                args.num_train_iter,
                                                num_warmup_steps=args.num_train_iter * 0)
    ## set SGD and cosine lr on FixMatch 
    model.set_optimizer(optimizer, scheduler)

    # SET Devices for (Distributed) DataParallel
    # if not torch.cuda.is_available():
    #     raise Exception('ONLY GPU TRAINING IS SUPPORTED')
    # elif args.distributed:
    #     if args.gpu is not None:
    #         torch.cuda.set_device(args.gpu)

    #         '''
    #         batch_size: batch_size per node -> batch_size per gpu
    #         workers: workers per node -> workers per gpu
    #         '''
    #         args.batch_size = int(args.batch_size / ngpus_per_node)
    #         model.model.cuda(args.gpu)
    #         model.model = nn.SyncBatchNorm.convert_sync_batchnorm(model.model)
    #         model.model = torch.nn.parallel.DistributedDataParallel(model.model,
    #                                                                 device_ids=[args.gpu],
    #                                                                 broadcast_buffers=False,
    #                                                                 find_unused_parameters=True)

    #     else:
    #         # if arg.gpu is None, DDP will divide and allocate batch_size
    #         # to all available GPUs if device_ids are not set.
    #         model.cuda()
    #         model = torch.nn.parallel.DistributedDataParallel(model)

    # elif args.gpu is not None:
    #     print("CPU")
    #     # torch.cuda.set_device(args.gpu)
    #     # model.model = model.model.cuda(args.gpu)
    #     # model.model = model.model.cpu()

    # else:
    #     model.model = torch.nn.DataParallel(model.model).cuda()

    import copy
    model.ema_model = copy.deepcopy(model.model)

    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {args}")

    cudnn.benchmark = True
    # if args.rank != 0 and args.distributed:
    #     torch.distributed.barrier()
 
    # Construct Dataset & DataLoader
    if args.dataset != "imagenet":
        train_dset = SSL_Dataset(args, alg='fixmatch', name=args.dataset, train=True,
                                num_classes=args.num_classes, data_dir=args.data_dir)
        lb_dset, ulb_dset = train_dset.get_ssl_dset(args.num_labels)

        _eval_dset = SSL_Dataset(args, alg='fixmatch', name=args.dataset, train=False,
                                num_classes=args.num_classes, data_dir=args.data_dir)
        eval_dset = _eval_dset.get_dset()
    else:
        image_loader = ImageNetLoader(root_path=args.data_dir, num_labels=args.num_labels,
                                      num_class=args.num_classes)
        lb_dset = image_loader.get_lb_train_data()
        ulb_dset = image_loader.get_ulb_train_data()
        eval_dset = image_loader.get_lb_test_data()
    # if args.rank == 0 and args.distributed:
    #     torch.distributed.barrier()
                            
    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}

    loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                              args.batch_size,
                                              data_sampler=args.train_sampler,
                                              num_iters=args.num_train_iter,
                                              num_workers=args.num_workers)

    loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                               args.batch_size * args.uratio,
                                               data_sampler=args.train_sampler,
                                               num_iters=args.num_train_iter,
                                               num_workers=4 * args.num_workers)

    loader_dict['eval'] = get_data_loader(dset_dict['eval'],
                                          args.eval_batch_size,
                                          num_workers=args.num_workers,
                                          drop_last=False)

    ## set DataLoader on FixMatch
    model.set_data_loader(loader_dict)
    model.set_dset(ulb_dset)
    # If args.resume, load checkpoints from args.load_path
    if args.resume:
        model.load_model(args.load_path)

    # START TRAINING of FixMatch
    trainer = model.train
    for epoch in range(args.epoch):
        trainer(args, logger=logger)

    # if not args.multiprocessing_distributed or \
    #         (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
    #     model.save_model('latest_model.pth', save_path)
    model.save_model('latest_model.pth', save_path)

    # logging.warning(f"GPU {args.rank} training is FINISHED")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='fixmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('--use_tensorboard', action='store_true', help='Use tensorboard to plot and save curves, otherwise save the curves locally.')

    '''
    Training Configuration of FixMatch
    '''

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2 ** 20,
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=5000,
                        help='evaluation frequency')
    parser.add_argument('-nl', '--num_labels', type=int, default=40)
    parser.add_argument('-bsz', '--batch_size', type=int, default=64)
    parser.add_argument('--uratio', type=int, default=7,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')

    parser.add_argument('--hard_label', type=str2bool, default=True)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip', type=float, default=0)
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('-nc', '--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)

    '''
    multi-GPUs & Distrbitued Training
    '''

    # args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:11111', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=True,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    # config file
    parser.add_argument('--c', type=str, default='')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    main(args)
