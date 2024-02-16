import numpy as np

import torch
from torch import nn
from torchvision import models as torchvision_models

import Harmony.utils as utils
import Harmony.models.vision_transformer as vits
from Harmony.models.transformer import Transformer, LayerNorm
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
        self.context_length = 77
        self.vocab_size = 49408
        self.transformer_width = 512
        self.transformer_heads = 8 
        self.transformer_layers = 12
        self.embed_dim = 512
        self.vision_width = get_embedding_size_from_arch(self.meta['arch'])
        
        # define the model arch (i.e. dino, ibot, dino+mae, ibot+mae, harmony)
        self.define_arch() 

    def define_arch(self):

        # define the text encoder and peripherals
        self.text_encoder = Transformer(
            width=self.transformer_width,
            layers=self.transformer_layers,
            heads=self.transformer_heads,
            attn_mask=self.build_attention_mask(),
        ).cuda()
        self.image_projection = nn.Parameter(torch.empty(self.vision_width, self. embed_dim)).cuda()
        self.text_embedding = torch.nn.Embedding(self.vocab_size, self.transformer_width).cuda()
        self.text_positional_embedding = torch.nn.Parameter(torch.empty(self.context_length, self.transformer_width)).cuda()
        self.text_norm = LayerNorm(self.transformer_width).cuda()
        self.text_projection = torch.nn.Parameter(torch.empty(self.transformer_width, self.embed_dim)).cuda()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).cuda()
        
        self.initialize_contrastive_parameters()

        # define the image encoder(s)
        try: 
            self.image_encoder = vits.__dict__[self.meta['arch']](
                patch_size=self.meta['patch_size'],
                drop_path_rate=self.meta['drop_path_rate'],
                return_all_tokens=True if "ibot" in self.objective else False,
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
        self.is_discriminative = True
        self.is_generative = False
        self.is_contrastive = False 

        self.discriminative_path = DiscriminativePath(image_encoder=self.image_encoder, meta=self.meta).cuda()

        if "mae" in self.objective:
            self.generative_path = GenerativePath(backbone=self.gen_encoder, meta=self.meta).cuda()
            self.is_generative = True

            self.mask_ratio_scheduler = np.concatenate((
                np.linspace(self.meta['mask_ratio'],
                            self.meta['mask_ratio_end'], self.meta['mask_ratio_epochs'] * self.meta['num_iterations_per_epoch']),
                np.ones(self.meta['num_iterations_total'] -  (self.meta['mask_ratio_epochs'] * self.meta['num_iterations_per_epoch'])) * self.meta['mask_ratio']
            ))
            self.hard_labels_weight_scheduler = utils.cosine_scheduler(
                base_value=self.meta['hard_labels_weight'],
                final_value=self.meta['hard_labels_weight_end'],
                epochs=self.meta['epochs'],
                niter_per_ep=self.meta['num_iterations_per_epoch']
            )
        
        if "clip" in self.objective:
            self.contrastive_loss = CLIPLoss()
            self.is_contrastive = True

    def initialize_contrastive_parameters(self):
        nn.init.normal_(self.text_embedding.weight, std=0.02)
        nn.init.normal_(self.text_positional_embedding, std=0.01)

        proj_std = (self.text_encoder.width ** -0.5) * ((2 * self.text_encoder.layers) ** -0.5)
        attn_std = self.text_encoder.width ** -0.5
        fc_std = (2 * self.text_encoder.width) ** -0.5
        for block in self.text_encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.text_encoder.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def encode_text(self, text):
        x = self.text_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.text_positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.text_norm(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
            
    def forward(self, images, epoch, iteration, captions=None, masks=None):
        loss = torch.tensor([0.0]).to(self.meta['gpu'])
        outputs = {"loss": loss,
                   "disc_loss": torch.zeros(1),
                   "gen_loss": torch.zeros(1),
                   "clip_loss": torch.zeros(1)}
        
        if self.is_contrastive:
            text_embed = self.encode_text(captions)
            image_embed = self.image_encoder(images[1]) 
            image_embed = image_embed @ self.image_projection
            output = self.contrastive_loss(image_embed, text_embed, self.logit_scale.exp(), hard_weight=self.hard_labels_weight_scheduler[iteration])
            outputs["clip_loss"] = output['clip_loss']
            outputs["loss"] += output['clip_loss']

        if self.is_discriminative:
            output = self.discriminative_path(images[1:], epoch, masks) # first image is simply augmeneted image
            
            outputs["teacher_output"] = output["teacher_output"]
            outputs["teacher_output"] = output["student_output"]
            outputs["disc_loss"] = (output["loss"] * self.meta["disc_weight"])
            outputs["loss"] += (output["loss"] * self.meta["disc_weight"])

        if self.is_generative:
            output = self.generative_path(images, reconstruct_global_crops=self.meta['reconstruct_global_crops'], mask_ratio=self.mask_ratio_scheduler[epoch]) 
            
            outputs["pred"] = output["output"]
            outputs["mask"] = output["mask"]
            outputs["gen_loss"] = output["loss"]
            outputs["loss"] += output["loss"]

        return outputs