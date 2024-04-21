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
from Harmony.utils import DataAugmentation, get_dataset_from_string
from Harmony.models import Harmony
from Harmony.data import IBOTLoaderWrapper
from torch.cuda.amp.grad_scaler import OptState


torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('Harmony', add_help=False)

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
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--patch_out_dim', default=8192, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--shared_head', default=False, type=utils.bool_flag, help="""Wether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, help="""See above.
        Only works for teacher model.module. (Defeault: True)""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
        help="Norm to use in head for discriminative path (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
        help="Activation function in the projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--pred_ratio', default=0.3, type=float, nargs='+', help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
    parser.add_argument('--pred_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred_shape', default='block', type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    # parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
    #     help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--mask_ratio', default=0.75, type=float, help="Initial masking ratio for MAE.")
    parser.add_argument('--mask_ratio_end', default=0.75, type=float, help="Final value for masking ratio for MAE, after linear warmup.")
    parser.add_argument('--mask_ratio_epochs', default=10, type=int)
    parser.add_argument('--separate_gen_model', default=False, type=utils.bool_flag, help="""whether to separate the
        generative path""")
    parser.add_argument('--lambda1', default=0.5, type=float, help="""loss weight for dino
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=0.5, type=float, help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""")
    parser.add_argument('--objective', default='dino', type=str,
        choices=utils.power_set_permutations(['dino', 'ibot', 'mae', 'clip']),
        help="The method to use for training the model")

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
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--disc_weight', type=float, default=1, help="""Loss scaling for discriminative path""")
    parser.add_argument('--gen_weight', type=float, default=1, help="""Loss scaling for generative path""")
    parser.add_argument('--mlm_weight', type=float, default=1, help="""Loss scaling for mlm""")
    parser.add_argument('--text_dist_weight', type=float, default=1, help="""Loss scaling for text distillation""")
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
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
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
    parser.add_argument('--reconstruct_global_crops', type=utils.bool_flag, default=True, help="""Whether to reconstruct global crops or
                        entire image""")
    parser.add_argument('--contrastive_global_crops', type=utils.bool_flag, default=True, help="""Whether to use global crop in the 
                        contrastive learning objective""")
    parser.add_argument('--use_mlm', type=utils.bool_flag, default=False)
    parser.add_argument('--use_text_distillation', type=utils.bool_flag, default=False)
    parser.add_argument('--attentive_masking', type=utils.bool_flag, default=False)
    parser.add_argument('--random_masking', type=utils.bool_flag, default=False)
    parser.add_argument('--norm_pix_loss', type=utils.bool_flag, default=True)
    parser.add_argument('--hard_labels_weight', type=float, default=1.0, help="""Weight for using the hard labels in CLIP""")
    parser.add_argument('--hard_labels_weight_end', type=float, default=1.0, help="""Final value for hard labels weight in CLIP, after scheduler. 
                                                                                Same value as --hard_labels_weight for turning off soft clip loss.""")
    parser.add_argument('--hard_labels_weight_epochs', default=10, type=int)

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

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
    if "ibot" not in args.objective:
        args.use_masked_im_modeling = False

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ saving argument settings ... ============
    with open("{}/settings.pkl".format(args.output_dir), 'wb') as file:
        print("saving run setting")
        pickle.dump(dict(vars(args)), file)

    simple_aug = not(args.contrastive_global_crops and args.reconstruct_global_crops) 
    if not simple_aug:
        print("Not using simple augmentation to save memory")
    # ============ preparing data ... ============
    transform = DataAugmentation(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number, 
        args.local_crops_number,
        args.objective,
        simple_aug
    )
    data_root = args.data.split(":")[1]
    data = get_dataset_from_string(args.data)
    dataset = data(data_root, transform=transform)

    if "ibot" in args.objective:
        print("Using IBOT image transformation wrapper")
        dataset = IBOTLoaderWrapper(
            dataset=dataset,
            patch_size=args.patch_size,
            pred_ratio=args.pred_ratio,
            pred_ratio_var=args.pred_ratio_var,
            pred_aspect_ratio=(0.3, 1/0.3),
            pred_shape=args.pred_shape,
            pred_start_epoch=args.pred_start_epoch
        )
    
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

    model = Harmony(args=args, meta_training_data=meta_training_data).to(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(model)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups, eps=1e-07, betas=(0.9, 0.99))  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scalers = None
    if args.use_fp16:
        fp16_scalers = {
            "clip_loss": torch.cuda.amp.GradScaler(),
            "gen_loss": torch.cuda.amp.GradScaler(),
            "disc_loss": torch.cuda.amp.GradScaler(),
            "mlm_loss": torch.cuda.amp.GradScaler(),
            "text_dist_loss": torch.cuda.amp.GradScaler()
            }

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        # fp16_scalers=fp16_scalers,
        disc_loss=model.module.discriminative_path.loss if model.module.is_discriminative else None
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        if args.objective == "ibot":
            data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch ... ============        
        train_stats, meta_training_data = train_one_epoch(model, data_loader, optimizer,
            lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scalers, args, meta_training_data)

        # ============ writing logs ... ============
        save_dict = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'disc_loss': model.module.discriminative_path.loss.state_dict() if model.module.is_discriminative else None,
        }

        # saving teacher vit separately
        try: # yes I am lazy
            main_vit = model.module.teacher.backbone.state_dict() # if it has ibot/dino head, remove it
        except:
            try:
                main_vit = model.module.teacher.state_dict()
            except:
                main_vit = model.module.student.state_dict()

        if model.module.text_teacher != None or model.module.text_student != None:
            try:
                main_text = model.module.text_teacher.backbone.state_dict() # if it has text dist head, remove it
            except:
                try:
                    main_text = model.module.text_teacher.state_dict()
                except:
                    main_text = model.module.text_student.state_dict()
        else:
            main_text = None           

        # if len(fp16_scalers) > 0:
        #    save_dict['fp16_scaler'] = fp16_scalers
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
                    fp16_scalers, args, meta_training_data=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    names_tq, params_tq, names_tk, params_tk = [], [], [], []
    if model.module.teacher != None:
        for name_q, param_q in model.module.student.named_parameters():
            names_q.append(name_q)
            params_q.append(param_q)
        for name_k, param_k in model.module.teacher.named_parameters():
            names_k.append(name_k)
            params_k.append(param_k)

    if model.module.text_teacher != None:
        for name_tq, param_tq in model.module.text_student.named_parameters():
            names_tq.append(name_tq)
            params_tq.append(param_tq)
        for name_tk, param_tk in model.module.text_teacher.named_parameters():
            names_tk.append(name_tk)
            params_tk.append(param_tk)

    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

    for it, data in enumerate(metric_logger.log_every(data_loader, 10, header)):

        if len(data) == 3:
            images, captions, masks = data
            masks = [m.cuda(non_blocking=True) for m in masks]
        elif len(data) == 2:
            images, captions = data
            masks = None
        
        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        captions = captions.cuda()

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # teacher and student forward passes + compute loss
        with torch.cuda.amp.autocast(args.use_fp16):
            losses, unscaled_soft_loss = model(images, epoch, it, masks=masks, captions=captions)
            loss = sum(losses.values()).item()

        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        student = model.module.student
        if not args.use_fp16:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            if model.module.is_discriminative:
                utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:   
            keys = list(fp16_scalers.keys())
            with torch.cuda.amp.autocast():
                scaled_loss = torch.tensor([0.0], device=args.gpu)
                for k, v in losses.items(): 
                    scaled_loss += fp16_scalers[k].scale(v) 
            scaled_loss.backward()

            # for k in keys[1:]: 
            #     fp16_scalers[k]._per_optimizer_states[id(optimizer)]['stage'] = OptState.UNSCALED 

            for k in losses.keys():
                fp16_scalers[k].unscale_(optimizer)

            # fp16_scalers[keys[0]].unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place

            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            if model.module.is_discriminative:
                utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
                
            fp16_scalers[keys[0]].step(optimizer)

            # for k in losses.keys():
            #     fp16_scalers[k]._check_inf_per_device(optimizer)
            
            for k in losses.keys():
                fp16_scalers[k].update()    

        # EMA update for the teacher
        m = momentum_schedule[it]  # momentum parameter
        if model.module.teacher != None:
            with torch.no_grad():
                for param_q, param_k in zip(params_q, params_k):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        
        if model.module.text_teacher != None:
            with torch.no_grad():
                for param_tq, param_tk in zip(params_tq, params_tk):
                    param_tk.data.mul_(m).add_((1 - m) * param_tq.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss)
        if 'disc_loss' in losses.keys(): metric_logger.update(discriminative_loss=losses["disc_loss"].item())
        if 'gen_loss' in losses.keys(): metric_logger.update(generative_loss=losses["gen_loss"].item())
        if 'clip_loss' in losses.keys(): metric_logger.update(clip_loss=losses["clip_loss"].item())
        if unscaled_soft_loss != 0: metric_logger.update(unscaled_soft_loss=unscaled_soft_loss)
        if 'mlm_loss' in losses.keys(): metric_logger.update(mlm_loss=losses['mlm_loss'].item())
        if 'text_dist_loss' in losses.keys(): metric_logger.update(text_dist_loss=losses['text_dist_loss'].item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        if model.module.is_contrastive: metric_logger.update(clip_hard_weight=model.module.hard_labels_weight_scheduler[it])
        if model.module.is_generative: metric_logger.update(mask_ratio=round(model.module.mask_ratio_scheduler[it], 2))    
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, meta_training_data
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('HARMONY', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
