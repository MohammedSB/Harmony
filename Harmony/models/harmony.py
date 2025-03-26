import numpy as np

import torch
from torch import nn
from torchvision import models as torchvision_models

import Harmony.utils as utils
import Harmony.models.vision_transformer as vits
from Harmony.models.transformer import Transformer, LayerNorm
from Harmony.models.text_decoder import TextDecoder
from .text_distillation import TextDistillationPath
from .generative import GenerativePath
from .discriminative import DiscriminativePath
from .contrastive import ContrastivePath
from .utils import get_embedding_size_from_arch, get_masked_captions, get_att_mask

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
       
        # initialize variables
        self.is_discriminative, self.is_generative, self.is_contrastive = False, False, False  
        self.teacher, self.student, self.text_teacher, self.text_student = None, None, None, None

        if "dino" in self.objective or "ibot" in self.objective:
            self.discriminative_path = DiscriminativePath(image_encoder=self.image_encoder, meta=self.meta)
            self.is_discriminative = True

            self.teacher = self.discriminative_path.teacher
            self.student = self.discriminative_path.student

        if "mae" in self.objective:
            self.generative_path = GenerativePath(backbone=self.image_encoder, meta=self.meta)
            self.is_generative = True

        if "clip" in self.objective:
            self.contrastive_path = ContrastivePath(image_backbone=self.image_encoder, meta=self.meta, use_soft_labels=self.use_soft_labels)
            self.is_contrastive = True

            self.text_student = self.contrastive_path.text_backbone
            if hasattr(self.contrastive_path, 'text_backbone_teacher'): # check if we already defined a text teacher from contrastive path 
                self.text_teacher = self.contrastive_path.text_backbone_teacher

            if self.meta['use_mlm']:
                print("Using masked language modeling")
                self.mlm_head = TextDecoder(transformer_layers=4)

            if self.meta['use_text_distillation']:
                print("Using text self-dist")
                self.text_distillation_path = TextDistillationPath(meta=self.meta, text_student=self.text_student, text_teacher=self.text_teacher)

                # expand student and teacher networks to include the text distillation heads 
                self.text_student = self.text_distillation_path.text_dist_student
                self.text_teacher = self.text_distillation_path.text_dist_teacher

        if self.student == None:
            self.student = self.image_encoder

        if (self.use_soft_labels and self.teacher == None) or (self.meta['attentive_masking'] and self.teacher == None):
            print("Defining a image teacher encoder for soft labels or attentive masking")
            self.teacher = vits.__dict__[self.meta['arch']](
                patch_size=self.meta['patch_size'],
                return_all_tokens=False,
                can_be_contrastive=True,
            )
            self.teacher.load_state_dict(self.student.state_dict(), strict=False)
            for p in self.teacher.parameters():
                p.requires_grad = False

    def forward_discriminative(self, images, epoch, iteration, captions=None, masks=None):
        output = self.discriminative_path(images[1:], epoch, masks=masks) # first image is simply augmeneted image
        loss = output["loss"] * self.meta["disc_weight"]
        return loss
            
    def forward_contrastive(self, images, epoch, iteration, captions=None, masks=None):
        # see which teacher model we can use, if any            
        teacher = None
        if self.is_discriminative:
            teacher = self.discriminative_path.teacher.backbone
        elif self.teacher != None:
            teacher = self.teacher
            
        teacher_attn = None # TODO: fix this
        unscaled_soft_loss = None

        hard_weight = self.hard_labels_weight_scheduler[iteration]
        output = self.contrastive_path(images, captions, hard_weight, teacher, teacher_attn)
        loss = output['clip_loss']
        if 'soft_loss' in output.keys(): unscaled_soft_loss = output['soft_loss'].item()
        return loss, unscaled_soft_loss
    
    def forward_text(self, images, epoch, iteration, captions=None, masks=None):
        labels = captions.detach().clone()
        masked_captions, labels, masks_c = get_masked_captions(captions=captions, labels=labels)
        _, text_embedding = self.contrastive_path.text_backbone(masked_captions, return_without_proj=True)
        loss = torch.tensor([0.0]).to(self.meta['gpu'])
        mlm_loss, dist_loss = None, None
                
        if self.meta['use_mlm']:
            mlm_output = self.mlm_head(text_embedding)
            
            probs = mlm_output.view(-1, mlm_output.size(-1)) 
            labels = labels.view(-1)

            mlm_loss = torch.nn.functional.cross_entropy(probs, labels)
            if torch.isnan(loss):
                mlm_loss = torch.tensor(0.0) 
            
            loss += mlm_loss * self.meta["mlm_weight"]
            mlm_loss = mlm_loss.item()

        if self.meta['use_text_distillation']:
            dist_loss = self.text_distillation_path(captions, masked_captions, masks_c, epoch, text_embedding)
            loss += dist_loss * self.meta["text_dist_weight"]
            dist_loss = dist_loss.item()
        
        return loss, mlm_loss, dist_loss

    def forward_generative(self, images, epoch, iteration, captions=None, masks=None):
        output = self.generative_path(images, reconstruct_global_crops=self.meta['reconstruct_global_crops'], mask_ratio=self.mask_ratio_scheduler[epoch]) 
        loss = output["loss"] * self.meta["gen_weight"]
        return loss
            
    def forward(self, images, epoch, iteration, captions=None, masks=None, path="d"):
        if path == "d":
            return self.forward_discriminative(images, epoch, iteration, captions=captions, masks=masks)
        elif path == "c":
            return self.forward_contrastive(images, epoch, iteration, captions=captions, masks=masks)
        elif path == "t":
            return self.forward_text(images, epoch, iteration, captions=captions, masks=masks)
        elif path == "g":
            return self.forward_generative(images, epoch, iteration, captions=captions, masks=masks)
        else:
            raise Exception("Please select one of the four paths")
