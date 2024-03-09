import numpy as np

import torch
from torch import nn
from torchvision import models as torchvision_models

import Harmony.utils as utils
import Harmony.models.vision_transformer as vits
from Harmony.models.transformer import Transformer, LayerNorm
from Harmony.models.text_encoder import TextEncoder
from Harmony.losses import CLIPLoss
from .generative import GenerativePath
from .discriminative import DiscriminativePath
from .utils import get_embedding_size_from_arch

class Harmony(torch.nn.Module):
    def __init__(self, args, meta_training_data=None):
        super().__init__()
        self.meta = vars(args)
        if meta_training_data != None:
            self.meta = {**self.meta, **meta_training_data}
        self.objective = args.objective

        # initialize text encoder properties (vit-b)
        self.vision_width = get_embedding_size_from_arch(self.meta['arch'])
        
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
            )
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
       
        # initialize the variables
        self.is_discriminative = False
        self.is_generative = False
        self.is_contrastive = False 

        if "dino" in self.objective or "ibot" in self.objective:
            self.discriminative_path = DiscriminativePath(image_encoder=self.image_encoder, meta=self.meta).cuda()
            self.is_discriminative = True

        if "mae" in self.objective:
            self.generative_path = GenerativePath(backbone=self.gen_encoder, meta=self.meta).cuda()
            self.generative_path =  nn.parallel.DistributedDataParallel(self.generative_path, device_ids=[self.meta['gpu']])
            self.is_generative = True

            self.mask_ratio_scheduler = np.concatenate((
                np.linspace(self.meta['mask_ratio'],
                            self.meta['mask_ratio_end'], self.meta['mask_ratio_epochs'] * self.meta['num_iterations_per_epoch']),
                np.ones(self.meta['num_iterations_total'] -  (self.meta['mask_ratio_epochs'] * self.meta['num_iterations_per_epoch'])) * self.meta['mask_ratio_end']
            ))

        if "clip" in self.objective:
            self.contrastive_loss = CLIPLoss()
            self.is_contrastive = True

            self.hard_labels_weight_scheduler = utils.cosine_scheduler(
                base_value=self.meta['hard_labels_weight'],
                final_value=self.meta['hard_labels_weight_end'],
                epochs=self.meta['epochs'],
                niter_per_ep=self.meta['num_iterations_per_epoch']
            )
            # define the text encoder and peripherals
            text_embed_dim = 512
            self.text_encoder = TextEncoder(embed_dim=text_embed_dim)
            self.text_encoder = nn.parallel.DistributedDataParallel(self.text_encoder, device_ids=[self.meta['gpu']])
            self.image_projection = nn.Parameter(torch.empty(self.vision_width, text_embed_dim)).cuda()
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).cuda()
            nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)

            # check if to whether to use labels 
            self.use_soft_labels = np.any(self.hard_labels_weight_scheduler < 1.0) 
            if self.use_soft_labels:
                print("Using soft labels!")
                self.image_projection_teacher = nn.Parameter(torch.empty(self.vision_width, text_embed_dim)).cuda()
                self.text_encoder_teacher = TextEncoder(embed_dim=text_embed_dim)
                self.text_encoder_teacher  = nn.parallel.DistributedDataParallel(self.text_encoder_teacher, device_ids=[self.meta['gpu']])
                for param in self.text_encoder_teacher.parameters():
                    param.requires_grad = False

        if not self.is_discriminative:
            self.image_encoder = self.image_encoder.cuda()
            
    def forward(self, images, epoch, iteration, captions=None, masks=None):
        loss = torch.tensor([0.0]).to(self.meta['gpu'])
        outputs = {"loss": loss,
                   "disc_loss": torch.zeros(1),
                   "gen_loss": torch.zeros(1),
                   "clip_loss": torch.zeros(1)}
        
        if self.is_contrastive:

            # TODO: do this in a better way
            self.image_encoder.masked_im_modeling = False
            self.image_encoder.return_all_tokens = False

            text_embed = self.text_encoder(captions)
            image_embed = self.image_encoder(images[1]) # input simply augmeneted image
            image_embed = image_embed @ self.image_projection
            hard_weight = self.hard_labels_weight_scheduler[iteration]

            if self.use_soft_labels and self.is_discriminative:
                # TODO: do this in a better way
                self.discriminative_path.teacher.backbone.return_all_tokens = False
                
                image_embed_teacher = self.discriminative_path.teacher.backbone(images[1]) 
                image_embed_teacher = image_embed_teacher @ self.image_projection_teacher
                text_embed_teacher =  self.text_encoder_teacher(captions)

                self.discriminative_path.teacher.backbone.return_all_tokens = self.meta["return_all_tokens"]
            else:
                image_embed_teacher = None
                text_embed_teacher = None
            output = self.contrastive_loss(image_embed, text_embed, self.logit_scale.exp(),
                                            image_embed_teacher=image_embed_teacher,
                                            text_embed_teacher=text_embed_teacher,
                                            hard_weight=hard_weight)
            
            self.image_encoder.masked_im_modeling = self.meta['use_masked_im_modeling']
            self.image_encoder.return_all_tokens = self.meta["return_all_tokens"]

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