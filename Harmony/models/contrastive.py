import numpy as np

import torch
import torch.nn as nn

from Harmony.models.vision_transformer import Block
from Harmony.models.text_encoder import TextEncoder
from Harmony.losses import CLIPLoss
from .utils import get_att_mask, get_att_mask_2

class ContrastivePath(nn.Module):
    def __init__(self, image_backbone, meta, use_soft_labels):
        super().__init__()
        self.image_backbone = image_backbone
        self.meta = meta
        self.use_soft_labels = use_soft_labels
        
        self.loss = self.contrastive_loss = CLIPLoss()

        # define the text encoder and peripherals
        text_embed_dim = 512
        vocab_size = 49409 # 49408 + mask id
        self.text_backbone = TextEncoder(embed_dim=text_embed_dim, vocab_size=vocab_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # check if to whether to use labels 
        if self.use_soft_labels:
            print("Using soft labels!")
            self.text_backbone_teacher = TextEncoder(embed_dim=text_embed_dim, vocab_size=vocab_size)
            self.text_backbone_teacher.load_state_dict(self.text_backbone.state_dict(), strict=False)
            for param in self.text_backbone_teacher.parameters():
                param.requires_grad = False

        
    def forward(self, images, captions, hard_weight, teacher=None, teacher_attn=None):
        # TODO: do this in a better way
        self.image_backbone.masked_im_modeling = False
        self.image_backbone.return_all_tokens = False

        indx = int(self.meta['contrastive_global_crops']) 

        if self.meta['attentive_masking']:
            if teacher_attn != None:
                attention_map = teacher_attn
                attention_map = attention_map[0::2]
            else:
                _, attention_map = teacher(images[indx], return_attn=True)
            mask = get_att_mask_2(attention_map, ratio=0.5)
            remove_mask = True 
        else:
            mask = None
            remove_mask = False

        text_embed = self.text_backbone(captions)
        image_embed = self.image_backbone(images[indx], mask=mask, remove_mask=remove_mask, contrastive=True) 

        if self.use_soft_labels and teacher:
            # TODO: do this in a better way
            teacher.return_all_tokens = False
            
            image_embed_teacher = teacher(images[indx], contrastive=True) 
            text_embed_teacher =  self.text_backbone_teacher(captions)

            teacher.return_all_tokens = self.meta["return_all_tokens"]
        else:
            image_embed_teacher = None
            text_embed_teacher = None
        output = self.contrastive_loss(image_embed, text_embed, self.logit_scale.exp(),
                                        image_embed_teacher=image_embed_teacher,
                                        text_embed_teacher=text_embed_teacher,
                                        hard_weight=hard_weight)

        self.image_backbone.masked_im_modeling = self.meta['use_masked_im_modeling']
        self.image_backbone.return_all_tokens = self.meta["return_all_tokens"]

        return output   