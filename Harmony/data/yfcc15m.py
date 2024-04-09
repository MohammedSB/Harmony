import os
import torch

import pandas as pd
import numpy as np
from Harmony.data.utils import SimpleTokenizer
from PIL import Image

def get_files_from_root(directory):
    f = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            f.append(file_path.split(os.sep)[-1].split(".")[0])
    return f

class YFCC15M(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, tokneizer=SimpleTokenizer(), **kwargs):
        self.root = root
        
        files = get_files_from_root(self.root + os.sep + "images")
        self.df = pd.read_csv(root + os.sep + "yfcc15m.csv")
        indices = self.df['1'].isin(files)
        self.df = self.df[indices]
        
        self.image_captions = [tuple(x[2:]) for x in self.df.to_numpy()]
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