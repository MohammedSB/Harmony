import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
from utils import DataAugmentationDINO, get_dataset_from_string
import vision_transformer as vits
from .generative import GenerativePath
from .discriminative import DiscriminativePath

class Harmony(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.meta = vars(args)
        self.objective = args.objective

        # define the model arch (i.e. dino, dino+mae, harmony)
        self.define_arch()    

    def define_arch(self):

        # define the image encoder
        try: 
            self.meta['arch'] in vits.__dict__.keys()
            self.image_encoder = vits.__dict__[self.meta['arch']](
                patch_size=self.meta['patch_size'],
                drop_path_rate=self.meta['drop_path_rate'],  # stochastic depth
            )            
        except:
            raise Exception(f"Unknow arch: {self.meta['arch']}")

        self.meta['embed_dim'] = self.image_encoder.embed_dim
       
        if "dino" in self.objective:
            self.discrimitavie_path = DiscriminativePath(image_encoder=self.image_encoder, meta=self.meta)
            self.is_discriminative = True
        if "mae" in self.objective:
            self.generative_path = GenerativePath(image_encoder=self.image_encoder)
            self.is_generative = True


    def forward(self, images, epoch):
        if self.is_discriminative:
            disc_output = self.discrimitavie_path(images, epoch)
            teacher_output = disc_output["teacher_output"]
            student_output = disc_output["student_output"]
            disc_loss = disc_output["loss"]

        return {
            "teacher_output": teacher_output,
            "student_output": student_output,
            "disc_loss": disc_loss,
        }