import os
import torch

from tqdm import tqdm
import pandas as pd
import numpy as np
from Harmony.data.utils import SimpleTokenizer
from PIL import Image
from multiprocessing import Manager
import pickle

import csv

class YFCC15M(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, tokneizer=SimpleTokenizer(), **kwargs):
        self.root = root
        manager = Manager()
        # files = get_files_from_root(self.root + os.sep + "images")
        with open(root + os.sep + "yfcc15m_2.pkl", 'rb') as f:
            self.image_captions = manager.list(pickle.load(f))
        #f = open('/ibex/user/baharoms/t.csv', 'r')
        #c = csv.reader(f)
        #next(c)
        #l = []
        #for row in c:
        #    l.append(c)
        #self.image_captions = [print(x[0]) for x in c if len(x) > 0 and not x[0].isdigit()]
        #f.close()
        self.transform = transform
        self.tokenizer = tokneizer
        print("Number of images loaded in YFCC15M are:", {self.__len__()})

    def __len__(self):
        return len(self.image_captions)

    def get_image_caption_pair(self, idx):
        item = self.image_captions[idx]
        path = item[1]
        path = self.root + os.sep + "images" + os.sep + path[:3] + os.sep + path[3:6] + os.sep + path + ".jpg"
        image = Image.open(path).convert("RGB")
        caption = np.random.choice([item[2], item[3]])
        return image, caption


    def __getitem__(self, idx):
        image, caption = self.get_image_caption_pair(idx)
        if self.transform:
            image = self.transform(image)
        if self.tokenizer:
            caption = self.tokenizer(caption)

        torch.cuda.empty_cache()
        return image, caption