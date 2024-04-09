import os
import torch

import pandas as pd
import numpy as np
from Harmony.data.utils import SimpleTokenizer
from PIL import Image

class YFCC15M(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, tokneizer=SimpleTokenizer(), **kwargs):
        self.root = root
        self.folders =  [f for f in os.scandir(root) if f.is_dir()]
        self.image_captions = pd.read_csv(root + os.sep + "yfcc15m.csv")
        self.image_captions = [tuple(x[2:]) for x in self.image_captions.to_numpy()]
        self.transform = transform
        self.tokenizer = tokneizer
        print("Number of images loaded in YFCC15M are:", {self.__len__()})

    def __len__(self):
        return len(self.image_captions)

    def get_image_caption_pair(self, idx):
        item = self.image_captions[0]
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