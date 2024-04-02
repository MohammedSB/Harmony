import numpy as np

import torch
from torch import nn
from torchvision import models as torchvision_models

import Harmony.utils as utils
import Harmony.models.vision_transformer as vits
from Harmony.models.transformer import Transformer, LayerNorm
from Harmony.models.text_encoder import TextEncoder
from Harmony.models.text_decoder import TextDecoder
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
                masked_im_modeling=self.meta['use_masked_im_modeling'],
                can_be_contrastive=True,
            )
            # if self.meta['separate_gen_model']:
            #     print("Building separate network for generative path.")
            #     self.gen_encoder = vits.__dict__[self.meta['arch']](
            #         patch_size=self.meta['patch_size'],
            #         drop_path_rate=self.meta['drop_path_rate'] if hasattr(self.meta, 'drop_path_rate') else 0,  # stochastic depth
            #     )
            # else:
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

            if self.meta['use_mlm']:
                self.mlm_head = TextDecoder(transformer_layers=4)

        if self.use_soft_labels and not self.is_discriminative:
            self.teacher = vits.__dict__[self.meta['arch']](
                patch_size=self.meta['patch_size'],
                return_all_tokens=True if "ibot" in self.meta['objective'] else False,
                can_be_contrastive=True,
            )
            
    def forward(self, images, epoch, iteration, captions=None, masks=None):
        loss = torch.tensor([0.0]).to(self.meta['gpu'])
        outputs = {"loss": loss}
        
        if self.is_contrastive:
            if self.use_soft_labels:
                teacher = self.discriminative_path.teacher.backbone if self.is_discriminative else self.teacher
                hard_weight = self.hard_labels_weight_scheduler[iteration] if self.is_discriminative else 0
                output = self.contrastive_path(images, captions, hard_weight, teacher)
                if 'soft_loss' in output.keys(): outputs['soft_loss'] = output['soft_loss'].item()
            else:
                output = self.contrastive_path(images, captions, self.hard_labels_weight_scheduler[iteration])
            outputs["clip_loss"] = output['clip_loss'].item()
            outputs["loss"] += output['clip_loss']

            if self.meta['use_mlm']:
                labels = captions.detach().clone()
                masks_c = torch.ones_like(captions) * 0.20
                masks_c = torch.bernoulli(masks_c)

                # zero out all values past the eot.
                argmax_indices = captions.argmax(dim=1)
                range_tensor = torch.arange(captions.shape[1], device=captions.device).expand_as(captions)
                condition_mask = range_tensor > argmax_indices.unsqueeze(1)
                condition_mask[:, 0:1] = 1 # make all sot zero. 1 means add it for condition to remove. 
                row_indices = torch.arange(condition_mask.size(0))
                condition_mask[row_indices, argmax_indices] = 1 # make all eot zero.
                masks_c[condition_mask] = 0 # make sot and eot zero.

                # get masked input captions
                masked_captions = captions.clone()
                masked_captions[masks_c == 1] = 49408 # This is mask ID. TODO: do not hard code this
                labels[masks_c == 0] = -100 # this is the default ignore value for pytorch ce

                _, mlm_output = self.contrastive_path.text_backbone(masked_captions, return_without_proj=True)
                mlm_output = self.mlm_head(mlm_output)
                
                probs = mlm_output.view(-1, mlm_output.size(-1)) 
                labels = labels.view(-1)

                loss = torch.nn.functional.cross_entropy(probs, labels)
                
                outputs["mlm_loss"] = loss.item()
                outputs["loss"] += loss

        if self.is_discriminative:
            output = self.discriminative_path(images[1:], epoch, masks=masks) # first image is simply augmeneted image

            outputs["disc_loss"] = output["loss"].item() * self.meta["disc_weight"]
            outputs["loss"] += (output["loss"] * self.meta["disc_weight"])

        if self.is_generative:
            output = self.generative_path(images, reconstruct_global_crops=self.meta['reconstruct_global_crops'], mask_ratio=self.mask_ratio_scheduler[epoch]) 
     
            outputs["gen_loss"] = output["loss"].item() * self.meta["gen_weight"]
            outputs["loss"] += (output["loss"] * self.meta["gen_weight"])

        return outputs