# Based on ProtoCLIP code bases
# https://github.com/megvii-research/protoclip

import torch
import logging
import numpy as np
import tqdm
import os
from torchvision.datasets.coco import CocoCaptions
from torch.utils.data import Dataset, DataLoader
from Harmony.data.datasets import get_dataset_from_string
from Harmony.data.datasets import dataset_classes
import Harmony.models.vision_transformer as vits
from Harmony.models.text_encoder import TextEncoder
from Harmony.data.utils import SimpleTokenizer
import torchvision.transforms as transforms
from PIL import Image
from glob import glob
import argparse

def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


def get_args_parser():
    parser = argparse.ArgumentParser(description='SLIP 0-shot evaluations', add_help=False)
    parser.add_argument('--output_dir', default='./', type=str, help='output dir')
    parser.add_argument('--batch_size', default=2, type=int, help='batch_size')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--arch', default='vit_small')
    parser.add_argument('--patch_size', default=16)
    parser.add_argument('--flickr_data_dir', default='', type=str)
    parser.add_argument('--coco_data_dir', default='', type=str)
    parser.add_argument('--distributed', default=False, type=bool)
    return parser

class FlickDataset(Dataset):
    def __init__(
        self, flick_root=None, transform=None, tokenizer=None
    ):
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.transform = transform

        dataset_path = flick_root
        target_files = os.path.join(dataset_path,"results_20130124.token")
        captions = self.Read_Captions(target_files)

        img_path = glob(dataset_path + os.sep + "*jpg")
        txt_id = 0
        for index in range(len(img_path)):
            path = img_path[index]
            name = os.path.basename(path)
            self.image.append(path)
            self.img2txt[index] = []

            target = captions[name]
            
            for i, caption in enumerate(target):
                if tokenizer is None:
                    self.text.append(SimpleTokenizer(caption))
                else:
                    self.text.append(torch.stack([tokenizer(caption)]))
                self.img2txt[index].append(txt_id)
                self.txt2img[txt_id] = index
                txt_id += 1

        self.text = torch.cat(self.text, dim=0)

    def Read_Captions(self, Captions_Path):
        # here we will read caption file, and create a dictionary will hold the img name
        # as key and captions as value
        file = open(Captions_Path, "r", encoding="utf-8")
        Captions = file.read()
        file.close()
        Img_Captions_Dict = {}
        # now loop over the file and split each line with \n
        for Line in Captions.split("\n"):
            # each read line make tab split
            Line_Splitted = Line.split("\t")
            if len(Line_Splitted) < 2:
                continue
            Image_Name = Line_Splitted[0][
                :-2
            ]  # we make [:-2] because each img name followed by #1 or #2 or #3 so we want to remove this
            Caption_to_Img = Line_Splitted[1]
            if Image_Name not in Img_Captions_Dict:
                Img_Captions_Dict[Image_Name] = [Caption_to_Img]
            else:
                Img_Captions_Dict[Image_Name].append(Caption_to_Img)

        return Img_Captions_Dict

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = self.image[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, index


class CocoTexts:
    def __init__(self, coco_dataset):
        self.coco_dataset = coco_dataset

    def __len__(self):
        return len(self.coco_dataset.text)

    def __getitem__(self, index):
        return self.coco_dataset.text[index]


class CocoDataset(Dataset):
    # modeified from https://github.com/uta-smile/TCL/blob/main/dataset/caption_dataset.py#L50
    # get the ground truth (1 image v.s. multiple captions, hiting each of them is ok) for retrieval
    def __init__(
        self, coco_dataset=None, coco_val_root=None, transform=None, tokenizer=None
    ):
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.transform = transform

        txt_id = 0
        for index in range(len(coco_dataset)):
            ann_ids = coco_dataset.coco.getAnnIds(imgIds=coco_dataset.ids[index])
            anns = coco_dataset.coco.loadAnns(ann_ids)
            target = [ann["caption"] for ann in anns]

            path = coco_dataset.coco.loadImgs(coco_dataset.ids[index])[0]["file_name"]
            path = os.path.join(coco_val_root, path)

            self.image.append(path)
            self.img2txt[index] = []

            for i, caption in enumerate(target):
                if tokenizer is None:
                    self.text.append(SimpleTokenizer(caption))
                else:
                    self.text.append(torch.stack([tokenizer(caption)]))
                self.img2txt[index].append(txt_id)
                self.txt2img[txt_id] = index
                txt_id += 1
        self.text = torch.cat(self.text, dim=0)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = self.image[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, index


class FlickrTexts:
    def __init__(self, flickr_dataset):
        self.flickr_dataset = flickr_dataset

    def __len__(self):
        return len(self.flickr_dataset.text)

    def __getitem__(self, index):
        return self.flickr_dataset.text[index]


def flickr_retrieval_evaluation(image_encoder, text_encoder, preprocess, tokenizer, args):
    flickr_dataset = FlickDataset(args.flickr_data_dir, transform=preprocess, tokenizer=tokenizer)
    flickr_retrieval_dataloader = DataLoader(
        flickr_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    flickr_dataset_text = FlickrTexts(flickr_dataset)
    flickr_retrieval_text_dataloader = DataLoader(
        flickr_dataset_text,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    with torch.no_grad():
        print("extracting flickr text features...")
        all_text_features = []
        for texts in tqdm.tqdm(flickr_retrieval_text_dataloader):
            # texts = texts.to(args.device)
            texts = texts.cuda()
            if args.distributed and not args.horovod:
                text_features = text_encoder(texts).detach().cpu()
            else:
                text_features = text_encoder(texts).detach().cpu()
            all_text_features.append(text_features)
        all_text_features = torch.cat(all_text_features, dim=0)

        print("extracting flickr image features...")
        all_image_features = []
        for images, img_id in tqdm.tqdm(flickr_retrieval_dataloader):
            # images = images.to(args.device)
            images = images.cuda()

            if args.distributed and not args.horovod:
                image_features = image_encoder(images, contrastive=True).detach().cpu()
            else:
                image_features = image_encoder(images, contrastive=True).detach().cpu()

            all_image_features.append(image_features)
        all_image_features = torch.cat(all_image_features, dim=0)

        # normalization, this step is important
        all_image_features = all_image_features / all_image_features.norm(
            dim=-1, keepdim=True
        )
        all_text_features = all_text_features / all_text_features.norm(
            dim=-1, keepdim=True
        )

        scores_img2text = (all_image_features @ all_text_features.t()).detach()
        scores_text2img = scores_img2text.t().detach()

    retrieval_metrics = get_retrieval_metrics(
        scores_img2text.cpu().numpy(),
        scores_text2img.cpu().numpy(),
        flickr_retrieval_dataloader.dataset.img2txt,
        flickr_retrieval_dataloader.dataset.txt2img,
    )
    print("flickr retrieval evaluation: " + str(retrieval_metrics))

    deduplicated_text_features = torch.zeros_like(all_image_features)
    for i in range(len(flickr_retrieval_dataloader.dataset.img2txt)):
        deduplicated_text_features[i] = all_text_features[
            flickr_retrieval_dataloader.dataset.img2txt[i][0]
        ]

    return retrieval_metrics, all_image_features, deduplicated_text_features


def coco_retrieval_evaluation(model, preprocess, tokenizer, args):

    coco_val_root = os.path.join(args.coco_data_dir, "val2017")
    coco_val_json = os.path.join(
        args.coco_data_dir, "annotations/captions_val2017.json"
    )

    coco_dataset = CocoCaptions(
        root=coco_val_root, annFile=coco_val_json, transform=preprocess
    )
    coco_dataset = CocoDataset(
        coco_dataset,
        coco_val_root=coco_val_root,
        transform=preprocess,
        tokenizer=tokenizer,
    )
    coco_retrieval_dataloader = DataLoader(
        coco_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    coco_dataset_text = CocoTexts(coco_dataset)
    coco_retrieval_text_dataloader = DataLoader(
        coco_dataset_text,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    with torch.no_grad():
        print("extracting COCO text features...")
        all_text_features = []
        for texts in tqdm.tqdm(coco_retrieval_text_dataloader):
            # texts = texts.to(args.device)
            texts = texts.cuda()
            if args.distributed and not args.horovod:
                text_features = get_model(model).encode_text(texts).detach().cpu()
            else:
                text_features = get_model(model).encode_text(texts).detach().cpu()
            all_text_features.append(text_features)
        all_text_features = torch.cat(all_text_features, dim=0)

        print("extracting COCO image features...")
        all_image_features = []
        for images, img_id in tqdm.tqdm(coco_retrieval_dataloader):
            # images = images.to(args.device)
            images = images.cuda()

            if args.distributed and not args.horovod:
                image_features = get_model(model).encode_image(images).detach().cpu()
            else:
                image_features = get_model(model).encode_image(images).detach().cpu()

            all_image_features.append(image_features)
        all_image_features = torch.cat(all_image_features, dim=0)

        # normalization, this step is important
        all_image_features = all_image_features / all_image_features.norm(
            dim=-1, keepdim=True
        )
        all_text_features = all_text_features / all_text_features.norm(
            dim=-1, keepdim=True
        )

        scores_img2text = (all_image_features @ all_text_features.t()).detach()
        scores_text2img = scores_img2text.t().detach()

    retrieval_metrics = get_retrieval_metrics(
        scores_img2text.cpu().numpy(),
        scores_text2img.cpu().numpy(),
        coco_retrieval_dataloader.dataset.img2txt,
        coco_retrieval_dataloader.dataset.txt2img,
    )
    print("COCO retrieval evaluation: " + str(retrieval_metrics))

    deduplicated_text_features = torch.zeros_like(all_image_features)
    for i in range(len(coco_retrieval_dataloader.dataset.img2txt)):
        deduplicated_text_features[i] = all_text_features[
            coco_retrieval_dataloader.dataset.img2txt[i][0]
        ]

    return retrieval_metrics, all_image_features, deduplicated_text_features


def get_retrieval_metrics(scores_img2text, scores_text2img, gt_img2text, gt_text2img):

    # Images->Text
    ranks = np.zeros(scores_img2text.shape[0])
    for index, score in enumerate(scores_img2text):
        inds = np.argsort(score)[::-1]
        rank = 1e20
        for i in gt_img2text[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    img2text_recall_at_1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    img2text_recall_at_5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    img2text_recall_at_10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_text2img.shape[0])
    for index, score in enumerate(scores_text2img):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == gt_text2img[index])[0][0]

    text2img_recall_at_1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    text2img_recall_at_5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    text2img_recall_at_10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (img2text_recall_at_1 + img2text_recall_at_5 + img2text_recall_at_10) / 3
    ir_mean = (text2img_recall_at_1 + text2img_recall_at_5 + text2img_recall_at_10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {
        "image2text-R@1": img2text_recall_at_1,
        "image2text-R@5": img2text_recall_at_5,
        "image2text-R@10": img2text_recall_at_10,
        #'image2text-R-mean': tr_mean,
        "text2image-R@1": text2img_recall_at_1,
        "text2image-R@5": text2img_recall_at_5,
        "text2image-R@10": text2img_recall_at_10,
        #'text2image-R-mean': ir_mean,
        "mean-recall": r_mean,
    }

    print(eval_result)

    for key, item in eval_result.items():
        eval_result[key] = float(item)
    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser('retrival evaluations', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from collections import OrderedDict
    import slip_models as models

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

    tokenizer = SimpleTokenizer()
    val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    import json

    r, _, _ = coco_retrieval_evaluation(model, val_transform, tokenizer, args)
    with open(args.output_dir + os.sep + "coco_results.json", "w") as out: 
        json.dump(r, out)

    r, _, _ = flickr_retrieval_evaluation(model, val_transform, tokenizer, args)
    with open(args.output_dir + os.sep + "flickr_results.json", "w") as out: 
        json.dump(r, out)

