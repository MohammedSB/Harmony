import numpy as np

import torch
from torch import nn
from torchvision import models as torchvision_models

import Harmony.utils as utils
import Harmony.models.vision_transformer as vits
from Harmony.models.transformer import Transformer, LayerNorm
from .generative import GenerativePath
from .discriminative import DiscriminativePath
from .contrastive import ContrastivePath
from .utils import get_embedding_size_from_arch

class Harmony(torch.nn.Module):
    def __init__(self, args, meta_training_data=None):
        super().__init__()
        self.meta = vars(args)
        if meta_training_data != None:
            self.meta = {**self.meta, **meta_training_data}
        self.objective = args.objective
        
        # define the model arch (i.e. dino, ibot, dino+mae, ibot+mae, harmony)
        self.define_arch() 

    def define_arch(self):
        # define the image encoder(s)
        try: 
            self.meta["return_all_tokens"] = True if "ibot" in self.objective else False
            self.image_encoder = vits.__dict__[self.meta['arch']](
                patch_size=self.meta['patch_size'],
                drop_path_rate=self.meta['drop_path_rate'],
                return_all_tokens=self.meta["return_all_tokens"],
                masked_im_modeling=self.meta['use_masked_im_modeling']
            ).cuda()
            if self.meta['separate_gen_model']:
                print("Building separate network for generative path.")
                self.gen_encoder = vits.__dict__[self.meta['arch']](
                    patch_size=self.meta['patch_size'],
                    drop_path_rate=self.meta['drop_path_rate'] if hasattr(self.meta, 'drop_path_rate') else 0,  # stochastic depth
                )
            else:
                self.gen_encoder = self.image_encoder
        except:
            raise Exception(f"Unknow arch: {self.meta['arch']}")
        self.meta['embed_dim'] = self.image_encoder.embed_dim

        self.mask_ratio_scheduler = np.concatenate((
                np.linspace(self.meta['mask_ratio'],
                self.meta['mask_ratio_end'], self.meta['mask_ratio_epochs'] * self.meta['num_iterations_per_epoch']),
                np.ones(self.meta['num_iterations_total'] -  (self.meta['mask_ratio_epochs'] * self.meta['num_iterations_per_epoch'])) * self.meta['mask_ratio_end']
            ))

        self.hard_labels_weight_scheduler = np.concatenate((utils.cosine_scheduler(
                base_value=self.meta['hard_labels_weight'],
                final_value=self.meta['hard_labels_weight_end'],
                epochs=self.meta['hard_labels_weight_epochs'],
                niter_per_ep=self.meta['num_iterations_per_epoch']
            ), np.ones(self.meta['num_iterations_total'] - (self.meta['hard_labels_weight_epochs'] * self.meta['num_iterations_per_epoch'])) * self.meta['hard_labels_weight_end']
        ))

        self.use_soft_labels = np.any(self.hard_labels_weight_scheduler < 1.0) 
       
        # initialize the variables
        self.is_discriminative = False
        self.is_generative = False
        self.is_contrastive = False 

        if "dino" in self.objective or "ibot" in self.objective:
            self.discriminative_path = DiscriminativePath(image_encoder=self.image_encoder, meta=self.meta)
            self.is_discriminative = True

        if "mae" in self.objective:
            self.generative_path = GenerativePath(backbone=self.gen_encoder, meta=self.meta)
            self.is_generative = True

        if "clip" in self.objective:
            self.contrastive_path = ContrastivePath(image_backbone=self.image_encoder, meta=self.meta, use_soft_labels=self.use_soft_labels)
            self.is_contrastive = True

        if self.use_soft_labels and not self.is_discriminative:
            self.teacher = vits.__dict__[self.meta['arch']](
                patch_size=self.meta['patch_size'],
                return_all_tokens=True if "ibot" in self.meta['objective'] else False,
            )
            
    def forward(self, images, epoch, iteration, captions=None, masks=None):
        loss = torch.tensor([0.0]).to(self.meta['gpu'])
        outputs = {"loss": loss,
                   "disc_loss": torch.zeros(1),
                   "gen_loss": torch.zeros(1),
                   "clip_loss": torch.zeros(1)}
        
        if self.is_contrastive:
            if self.use_soft_labels:
                teacher = self.discriminative_path.teacher.backbone if self.is_discriminative else self.teacher
                hard_weight = self.hard_labels_weight_scheduler[iteration] if self.is_discriminative else 0
                output = self.contrastive_path(images, captions, hard_weight, teacher)
                if 'soft_loss' in output.keys(): outputs['soft_loss'] = output['soft_loss']
            else:
                output = self.contrastive_path(images, captions, self.hard_labels_weight_scheduler[iteration])
            outputs["clip_loss"] = output['clip_loss']
            outputs["loss"] += output['clip_loss']

        if self.is_discriminative:
            output = self.discriminative_path(images[1:], epoch, masks=masks) # first image is simply augmeneted image
            
            outputs["teacher_output"] = output["teacher_output"]
            outputs["teacher_output"] = output["student_output"]
            outputs["disc_loss"] = output["loss"] * self.meta["disc_weight"]
            outputs["loss"] += (output["loss"] * self.meta["disc_weight"])

        if self.is_generative:
            output = self.generative_path(images, reconstruct_global_crops=self.meta['reconstruct_global_crops'], mask_ratio=self.mask_ratio_scheduler[epoch]) 
            
            outputs["pred"] = output["output"]
            outputs["mask"] = output["mask"]
            outputs["gen_loss"] = output["loss"] * self.meta["gen_weight"]
            outputs["loss"] += (output["loss"] * self.meta["gen_weight"])

        return outputs