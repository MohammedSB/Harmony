# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from collections import OrderedDict
import json
import os
from sklearn import metrics

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets_t
import inspect
import numpy as np
from PIL import Image

import slip_models as models
from Harmony.data.datasets import get_dataset_from_string
from Harmony.data.datasets import dataset_classes
import Harmony.models.vision_transformer as vits
from Harmony.models.text_encoder import TextEncoder
from Harmony.data.utils import SimpleTokenizer
from Harmony.data.meta.robust_mapping import indices_in_1k, thousand_k_to_200 

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class FileListDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(images)
        self.labels = np.load(labels)

    def __getitem__(self, index):
        img = pil_loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)

def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model

def get_args_parser():
    parser = argparse.ArgumentParser(description='SLIP 0-shot evaluations', add_help=False)
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--batch-size', default=256, type=int, help='batch_size')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    return parser


def main(args):
    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        ckpt_path = args.resume
    elif os.path.isfile(os.path.join(args.output_dir, 'checkpoint_best.pt')):
        ckpt_path = os.path.join(args.output_dir, 'checkpoint_best.pt')
    else:
        raise Exception('no checkpoint found')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    
    # create model
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    model = getattr(models, old_args.model)(rand_embed=False,
        ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
    model.cuda()
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

    cudnn.benchmark = True

    cwd = os.path.dirname(os.path.realpath(__file__))
    meta_dir = f"{os.sep}".join(cwd.split(f"{os.sep}")[:-1]) + f"{os.sep}data{os.sep}meta"
    with open(os.path.join(meta_dir, 'dataset_catalog.json')) as f:
        catalog = json.load(f)

    with open(os.path.join(meta_dir, 'templates.json')) as f:
        all_templates = json.load(f)

    with open(os.path.join(meta_dir, 'labels.json')) as f:
        all_labels = json.load(f)

    # Data loading code
    print("=> creating dataset")
    tokenizer = SimpleTokenizer()
    val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    results = []
    for d in catalog:
        dataset_name = d.upper()
        if dataset_name not in dataset_classes:
            continue
        print('Evaluating {}'.format(d))
        # val_dataset = datasets.get_downstream_dataset(catalog, name=d, is_train=False, transform=val_transform)
        entry = catalog[d]
        data_root = entry['path']

        if dataset_name == "IMAGENET":
            data = get_dataset_from_string(d + ":" + d)
            val_dataset = data(data_root + f"{os.sep}val", transform=val_transform, split="val")
        elif entry['type'] == 'imagefolder':
            val_dataset = datasets_t.ImageFolder(os.path.join(data_root, entry['test']), transform=val_transform)
        elif entry['type'] == 'torchvision':
            data = get_dataset_from_string(d + ":" + d)
            constructor = data.__init__
            signature = inspect.signature(constructor)
            parameters = signature.parameters
            if 'train' in parameters:
                val_dataset = data(data_root, transform=val_transform, train=False, download=True)
            elif 'split' in parameters:
                if d in ["patch_camelyon", "eurosat", "kitti_distance", "places"]:
                    val_dataset = data(data_root, transform=val_transform, split="val", download=True)
                else:
                    val_dataset = data(data_root, transform=val_transform, split="test", download=True)
            else:
                val_dataset = data(data_root, transform=val_transform, download=True)
        elif entry['type'] == 'filelist':
            path = entry['test']
            val_images = os.path.join(data_root, path + '_images.npy')
            val_labels = os.path.join(data_root, path + '_labels.npy')
            if d == 'clevr_counts':
                target_transform = lambda x: ['count_10', 'count_3', 'count_4', 'count_5', 'count_6', 'count_7', 'count_8', 'count_9'].index(x)
            else:
                target_transform = None
            val_dataset = FileListDataset(val_images, val_labels, val_transform, target_transform)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False)
        templates = all_templates[d]
        labels = all_labels[d]

        is_acc = d not in ['aircraft', 'pets', 'caltech101', 'flowers', 'kinetics700_frames', 'hateful_memes']

        acc_or_outputs = validate_zeroshot(val_loader, templates, labels, model, tokenizer, is_acc)

        if d in ['aircraft', 'pets', 'caltech101', 'flowers']:
            metric = mean_per_class(*acc_or_outputs)
        elif d == 'kinetics700_frames':
            top1, top5 = accuracy(*acc_or_outputs, topk=(1, 5))
            metric = (top1 + top5) / 2
            metric = metric.item()
        elif d == 'hateful_memes':
            metric = roc_auc(*acc_or_outputs)
        else:
            metric = acc_or_outputs

        results.append(metric)

        print('metric:', metric)

    print('all results:')
    for x in results:
        print('{:.1f}'.format(x))

def validate_zeroshot(val_loader, templates, labels, model, tokenizer, is_acc):
    # switch to evaluate mode
    model.eval()
    total_top1 = 0
    total_images = 0

    all_outputs = []
    all_targets = []

    print('=> encoding captions')
    with torch.no_grad():
        text_features = []
        for label in labels:
            if isinstance(label, list):
                texts = [t.format(l) for t in templates for l in label]
            else:
                texts = [t.format(label) for t in templates]
            texts = tokenizer(texts).cuda(non_blocking=True)
            texts = texts.view(-1, 77).contiguous()
            class_embeddings = get_model(model).encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        for images, target in val_loader:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # encode images
            image_features = get_model(model).encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_image = image_features @ text_features.t()

            if is_acc:
                # measure accuracy and record loss
                pred = logits_per_image.argmax(dim=1)
                correct = pred.eq(target).sum()
                total_top1 += correct.item()
                total_images += images.size(0)
            else:
                all_outputs.append(logits_per_image.cpu())
                all_targets.append(target.cpu())
            
    if is_acc:
        return 100 * total_top1 / total_images
    else:
        return torch.cat(all_outputs), torch.cat(all_targets)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_per_class(outputs, targets):
    pred = outputs.argmax(1)
    confusion_matrix = metrics.confusion_matrix(targets, pred)
    per_classes = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

    return 100 * per_classes.mean()


def roc_auc(outputs, targets):
    pos_score = outputs[:, 1] - outputs[:, 0]
    metric = metrics.roc_auc_score(targets, pos_score)

    return 100 * metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SLIP 0-shot evaluations', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
