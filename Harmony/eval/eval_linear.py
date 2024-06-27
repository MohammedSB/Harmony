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
import os
import sys
import argparse
import json
import pickle
import inspect
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import timm
from Harmony import utils
import Harmony.models.vision_transformer as vits
from Harmony.models import Harmony 
from Harmony.eval.metrics import MetricType, build_metric
from Harmony.data.datasets import get_dataset_from_string

def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ load args from settings file ... ============
    if args.settings_path != None:
        settings_path = args.settings_path
    else: # else look for the file in the pretrained weights dir
        settings_path = f"{os.sep}".join(args.pretrained_weights.split(os.sep)[:-1] + ["settings.pkl"])
    try:
        with open(settings_path, 'rb') as file:
            args_to_dump = pickle.load(file)
        for arg in args_to_dump.keys():
            if not hasattr(args, arg):
                setattr(args, arg, args_to_dump[arg])
    except:
        print("Settings file not found")

    # ============ building network ... ============
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    #model = timm.models.create_model(args.arch)
    model.cuda()
    model.eval()

    embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))

    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        args=args,
        n_last_blocks_list=[args.n_last_blocks],
        learning_rates=args.lrs,
        batch_size=args.batch_size_per_gpu,
        out_dim=embed_dim,
        num_classes=args.num_labels,
    )

    # linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    # linear_classifier = linear_classifier.cuda()
    # linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    data = get_dataset_from_string(args.data)
    data_path = args.data.split(':')[1] 

    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    sig = inspect.signature(data.__init__)
    if 'download' in sig.parameters:
        dataset_val = data(root=os.path.join(data_path, "val"), transform=val_transform, split="val", download=True)
    else:
        dataset_val = data(root=os.path.join(data_path, "val"), transform=val_transform, split="val")
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.evaluate:

        utils.load_pretrained_linear_weights(linear_classifiers, args.arch, args.patch_size)
        _, results_dict_temp = validate_network(args, val_loader, model, linear_classifiers, args.n_last_blocks, args.avgpool_patchtokens, num_classes=args.num_labels)

        results_dict = get_top_performer(results_dict_temp)

        print(f"Accuracy of the network on the {len(dataset_val)} test images: {results_dict['best_classifier']['accuracy']:.1f}%")
        return

    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    sig = inspect.signature(data.__init__)
    if 'download' in sig.parameters:
        dataset_train = data(root=os.path.join(data_path, "train"), transform=train_transform, download=True)
    else:
        dataset_train = data(root=os.path.join(data_path, "train"), transform=train_transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifiers,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, linear_classifiers, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            _, results_dict_temp = validate_network(args, val_loader, model, linear_classifiers, args.n_last_blocks, args.avgpool_patchtokens, num_classes=args.num_labels)
            best_classifier  = get_top_performer(results_dict_temp)
            best_cur_acc =  best_classifier['accuracy']['top-1']
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {best_cur_acc:.1f}%")
            best_acc = max(best_acc, best_cur_acc)
            print(f'Max accuracy so far: {best_acc:.2f}%')

            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in best_classifier['accuracy'].items()}}

        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifiers.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        outputs = linear_classifier(output)

        # compute cross entropy loss
        losses = {f"loss_{k}": nn.CrossEntropyLoss()(v, target) for k, v in outputs.items()}
        loss = sum(losses.values())
        # loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(args, val_loader, model, linear_classifier, n, avgpool, num_classes):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    linear_classifier = remove_ddp_wrapper(linear_classifier)
    postprocessors = {k: LinearPostprocessor(v, None) for k, v in linear_classifier.classifiers_dict.items()}

    metric = build_metric(MetricType.MEAN_ACCURACY, num_classes=num_classes)
    metrics = {k: metric.clone() for k in linear_classifier.classifiers_dict}

    for metric in metrics.values():
        metric = metric.cuda()

    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)

        # outputs = linear_classifier(output)
        
        # losses = {f"loss_{k}": nn.CrossEntropyLoss()(v, target) for k, v in output.items()}
        # loss = sum(losses.values())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](output, target)
            metric.update(**metric_inputs)

        # if linear_classifier.module.num_labels >= 5:
        #     acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        # else:
        #     acc1, = utils.accuracy(output, target, topk=(1,))

        # batch_size = inp.shape[0]
        # metric_logger.update(loss=loss.item())
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # if linear_classifier.module.num_labels >= 5:
        #     metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")
            
    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    for m in metrics.values():
        m.reset()

    # if linear_classifier.module.num_labels >= 5:
    #     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    # else:
    #     print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # del metrics
    return metric_logger_stats, stats


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)
    
class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)
    
def scale_lr(learning_rates, batch_size):
    return learning_rates * (batch_size * dist.get_world_size()) / 256.0


def setup_linear_classifiers(args, n_last_blocks_list, learning_rates, batch_size, out_dim=1024, num_classes=1000):
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for _lr in learning_rates:
        lr = scale_lr(_lr, batch_size)
        linear_classifier = LinearClassifier(
            dim=out_dim, num_labels=num_classes
        )
        linear_classifier = linear_classifier.cuda()
        linear_classifiers_dict[
            f"classifier_lr_{lr:.5f}".replace(".", "_")
        ] = linear_classifier
        optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    if dist.is_available():
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers, device_ids=[args.gpu])

    return linear_classifiers, optim_param_groups


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, class_mapping=None):
        super().__init__()
        self.linear_classifier = linear_classifier
        self.register_buffer("class_mapping", None if class_mapping is None else torch.LongTensor(class_mapping))

    def forward(self, samples, targets):
        preds = self.linear_classifier(samples)
        return {
            "preds": preds[:, self.class_mapping] if self.class_mapping is not None else preds,
            "target": targets,
        }
    
def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, nn.parallel.DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m

def get_top_performer(results_dict_temp):
    best_classifier_on_val = None
    print("")
    max_accuracy = 0
    best_classifier = ""
    for i, (classifier_string, metric) in enumerate(results_dict_temp.items()):
        print(f"-- Classifier: {classifier_string} * {metric}")
        if (
            best_classifier_on_val is None and metric["top-1"].item() >= max_accuracy
        ) or classifier_string == best_classifier_on_val:
            max_accuracy = metric["top-1"].item()
            best_classifier = classifier_string

    results_dict_temp[best_classifier] = {k: v.item() for k, v in results_dict_temp[best_classifier].items()}
    best_classifier = {"name": best_classifier, "accuracy": results_dict_temp[best_classifier]}
    return best_classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="main_vit", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lrs", default=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1], type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data', default='ImageNet:/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--settings_path', type=str, help='Path for the file that stores training run settings')
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    eval_linear(args)
