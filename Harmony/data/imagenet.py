import os

import numpy as np
from PIL import Image
import pandas as pd
import torch
from torchvision import datasets

class ImageNet(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transform=None, **kwargs):
        self.root = root
        self.transform = transform
        self.split = split

        if split == "train":
            self.data = datasets.ImageFolder(root, transform=transform)
            # self.data = torch.utils.data.Subset(self.data, range(10))
        elif split == "val":
            self.images =  os.listdir(self.root)
            self.image_paths = [os.path.join(self.root, image) for image in self.images]
            self.labels = pd.read_csv(f"{os.sep}".join(os.path.realpath(__file__).split(f"{os.sep}")[:-1]) + "/meta/imagenet_val_labels.csv")

            # self.image_paths = self.image_paths[:100]
            # self.images = self.images[:100]

    def __len__(self):
        if self.split == "train":
            return self.data.__len__()
        else:
            return len(self.images)

    def get_image_target(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image_name = self.images[idx].split('.')[0]
        label = self.labels[self.labels['ImageId'] == image_name]['PredictionString'].item()
        return image, label

    def __getitem__(self, idx):

        if self.split == "train":
            return self.data[idx][0], self.data[idx][1]
        
        image, target = self.get_image_target(idx)
        
        if self.transform:
            image = self.transform(image)

        return image, target