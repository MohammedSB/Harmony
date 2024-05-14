# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
import pickle
from pathlib import Path
from collections import OrderedDict
import gc

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import Harmony.utils as utils
from Harmony.utils import DataAugmentation
from Harmony.data.datasets import get_dataset_from_string
from Harmony.models import MaskCLIP
from Harmony.data import IBOTLoaderWrapper
from torch.cuda.amp.grad_scaler import OptState


torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('MaskCLIP', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.999, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
        help="Norm to use in head for discriminative path (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
        help="Activation function in the projection head (Default: gelu)")
    # parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
    #     help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--mask_ratio', default=0.75, type=float, help="Initial masking ratio for MAE.")
    parser.add_argument('--lambda1', default=1, type=float, help="""loss weight for dino
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1, type=float, help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=10, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--mask_dist_weight', type=float, default=0.05, help="""Loss scaling for mask self dist""")
    parser.add_argument('--mlm_weight', type=float, default=0.05, help="""Loss scaling for mlm""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--with_head', default=False, type=utils.bool_flag, help="whether to add IBOT ")

    # Misc
    parser.add_argument('--data', default='CC3M:/mnt/d/CC3M/cc3m', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train(args):

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ saving argument settings ... ============
    with open("{}/settings.pkl".format(args.output_dir), 'wb') as file:
        print("saving run setting")
        pickle.dump(dict(vars(args)), file)

    # ============ preparing data ... ============
    transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=3),
            transforms.ToTensor(),
            # transforms.RandomApply(
            #     [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            #     p=0.8
            # ),
            # utils.GaussianBlur(1.0),
            # transforms.RandomGrayscale(p=0.2),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    # transform = DataAugmentation(
    #     args.global_crops_scale,
    #     args.local_crops_scale,
    #     args.global_crops_number, 
    #     args.local_crops_number,
    #     args.objective,
    #     simple_aug
    # )
    
    data_root = args.data.split(":")[1]
    data = get_dataset_from_string(args.data)
    dataset = data(data_root, transform=transform)
    
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    meta_training_data = { 
        'num_iterations_per_epoch': len(data_loader),
        'num_iterations_total': len(data_loader) * args.epochs
    }

    model = MaskCLIP(args=args, meta_training_data=meta_training_data).to(args.gpu)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(model)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups, eps=1e-06, betas=(0.9, 0.98), weight_decay=0.1)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 0.9999. during training with a linear scheduler
    # following maskclip
    momentum_schedule = [
        args.momentum_teacher + (0.9999 - args.momentum_teacher) * i / (args.epochs * len(data_loader))
        for i in range(args.epochs * len(data_loader))]

    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        # disc_loss=model.discriminative_path.loss if model.is_discriminative else None
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        # data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch ... ============        
        train_stats, meta_training_data = train_one_epoch(model, data_loader, optimizer,
            lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args, meta_training_data)

        # ============ writing logs ... ============
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
 
        main_vit  = model.teacher.state_dict() 
        main_text = model.text_student.state_dict()
        
        if fp16_scaler != None:
           save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        utils.save_on_master(main_vit, os.path.join(args.output_dir, 'main_vit_checkpoint.pth'))
        if main_text != None:
            utils.save_on_master(main_text, os.path.join(args.output_dir, 'main_text_checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
            utils.save_on_master(main_vit, os.path.join(args.output_dir, f'main_vit_checkpoint{epoch:04}.pth'))
            if main_text != None:
                utils.save_on_master(main_text, os.path.join(args.output_dir, f'main_text_checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args, meta_training_data=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    if model.teacher != None:
        for name_q, param_q in model.student.named_parameters():
            names_q.append(name_q)
            params_q.append(param_q)
        for name_k, param_k in model.teacher.named_parameters():
            names_k.append(name_k)
            params_k.append(param_k)

    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

    for it, data in enumerate(metric_logger.log_every(data_loader, 10, header)):

        images, captions = data
    
        # move images to gpu
        # images = [im.cuda(non_blocking=True) for im in images]
        images = images.cuda()
        captions = captions.cuda()

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # teacher and student forward passes + compute loss
        with torch.cuda.amp.autocast(args.use_fp16):
            model_output = model(images, epoch, captions=captions)
            loss = model_output["loss"]

        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        student = model.student
        if not args.use_fp16:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad) # we should test clipping entire model.

            optimizer.step()
        else:   
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
    
            fp16_scaler.step(optimizer)
            fp16_scaler.update()  

        # EMA update for the teacher
        m = momentum_schedule[it]  # momentum parameter
        if model.teacher != None:
            with torch.no_grad():
                for param_q, param_k in zip(params_q, params_k):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        if 'mask_dist_loss' in model_output.keys(): metric_logger.update(mask_dist_loss=model_output["mask_dist_loss"])
        if 'clip_loss' in model_output.keys(): metric_logger.update(clip_loss=model_output["clip_loss"])
        if 'mlm_loss' in model_output.keys(): metric_logger.update(mlm_loss=model_output['mlm_loss'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, meta_training_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MaskCLIP', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
