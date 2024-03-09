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

from Harmony.utils import DataAugmentation, get_dataset_from_string
import Harmony.models.vision_transformer as vits
from Harmony.models.text_encoder import TextEncoder
from Harmony.data.utils import SimpleTokenizer
from Harmony.utils import dataset_classes

def get_args_parser():
    parser = argparse.ArgumentParser(description='SLIP 0-shot evaluations', add_help=False)
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--batch-size', default=32, type=int, help='batch_size')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--image_encoder', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--arch', default='vit_small')
    parser.add_argument('--patch_size', default=16)
    parser.add_argument('--text_encoder', default='', type=str, help='path to latest checkpoint')
    return parser


def main(args):
    # create model(s)
    image_encoder = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    image_encoder.cuda()
    image_state_dict = torch.load(args.image_encoder)
    image_state_dict = {k.replace("module.", ""): v for k, v in image_state_dict.items()}
    image_encoder.load_state_dict(image_state_dict, strict=True)
    print("=> loaded image checkpoint '{}'".format(args.image_encoder))

    
    text_encoder = TextEncoder(embed_dim=512)
    text_encoder.cuda()
    text_state_dict = torch.load(args.text_encoder)
    text_state_dict = {k.replace("module.", ""): v for k, v in text_state_dict.items()}
    text_encoder.load_state_dict(text_state_dict, strict=True)
    print("=> loaded text checkpoint '{}'".format(args.text_encoder))

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
        if d.upper() not in dataset_classes:
            continue
        print('Evaluating {}'.format(d))
        # val_dataset = datasets.get_downstream_dataset(catalog, name=d, is_train=False, transform=val_transform)
        data_root = catalog[d]['path']
        # data_root = args.data.split(":")[1]
        data = get_dataset_from_string(d + ":" + d)
        val_dataset = data(data_root, transform=val_transform)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False)

        templates = all_templates[d]
        labels = all_labels[d]

        is_acc = d not in ['aircraft', 'pets', 'caltech101', 'flowers', 'kinetics700_frames', 'hateful_memes']

        acc_or_outputs = validate_zeroshot(val_loader, templates, labels, image_encoder, text_encoder, tokenizer, is_acc)

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

def validate_zeroshot(val_loader, templates, labels, image_encoder, text_encoder, tokenizer, is_acc):
    # switch to evaluate mode
    image_encoder.eval()
    text_encoder.eval()
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
            class_embeddings = text_encoder(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        for images, target in val_loader:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # encode images
            image_features = image_encoder(images, contrastive=True)
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
