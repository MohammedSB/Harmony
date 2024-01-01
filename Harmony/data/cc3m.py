import os

from PIL import Image
import torch

def save_image_captions_from_folders(folders, root):
    images_paths  = []
    captions_path = []
    for folder in folders:
        for filename in os.listdir(folder):
            if ".jpg" in filename or ".png" in filename:
                images_paths.append(root + os.sep + folder.name + os.sep + filename)
            elif ".txt" in filename:
                captions_path.append(root + os.sep + folder.name + os.sep + filename)
    return images_paths, captions_path

class CC3M(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, **kwargs):
        self.root = root
        self.folders =  [f for f in os.scandir(root) if f.is_dir()]
        self.images, self.captions = save_image_captions_from_folders(self.folders, self.root)
        self.transform = transform

        assert len(self.captions) == len(self.images)
        print("Number of images loaded in CC3M are:", {self.__len__()})

    def __len__(self):
        return len(self.images)

    def get_image_caption_pair(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        caption_file = open(self.captions[idx])
        caption = caption_file.read()
        caption_file.close()

        return image, caption


    def __getitem__(self, idx):
        image, caption = self.get_image_caption_pair(idx)
        if self.transform:
            image = self.transform(image)

        return image, caption