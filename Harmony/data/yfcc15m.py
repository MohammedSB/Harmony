import os
import torch

from tqdm import tqdm
import pandas as pd
import numpy as np
from Harmony.data.utils import SimpleTokenizer
from PIL import Image

import csv

def get_files_from_root(directory):
    f = []
    for root, dirs, files in tqdm(os.walk(directory)):
        for file in files:
            file_path = os.path.join(root, file)
            f.append(file_path.split(os.sep)[-1].split(".")[0])
    return f

class YFCC15M(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, tokneizer=SimpleTokenizer(), **kwargs):
        self.root = root

        # files = get_files_from_root(self.root + os.sep + "images")
        f = open(root + os.sep + "yfcc15m_2.csv")
        c = csv.reader(f)
        next(c) # skip header

        self.image_captions = [tuple(x[-3:]) for x in c]
        f.close()
        self.transform = transform
        self.tokenizer = tokneizer
        print("Number of images loaded in YFCC15M are:", {self.__len__()})

    def __len__(self):
        return len(self.image_captions)

    def get_image_caption_pair(self, idx):
        item = self.image_captions[idx]
        path = item[0]
        path = self.root + os.sep + "images" + os.sep + path[:3] + os.sep + path[3:6] + os.sep + path + ".jpg"
        image = Image.open(path).convert("RGB")
        caption = np.random.choice([item[1], item[2]])
        return image, caption


    def __getitem__(self, idx):
        image, caption = self.get_image_caption_pair(idx)
        if self.transform:
            image = self.transform(image)
        if self.tokenizer:
            caption = self.tokenizer(caption)
    
        return image, caption