import numpy as np

import torch
from torch import nn
from torchvision import models as torchvision_models

import Harmony.utils as utils
import Harmony.models.vision_transformer as vits
from Harmony.models.transformer import Transformer, LayerNorm
from Harmony.models.text_decoder import TextDecoder
from .vision_transformer import VisionTransformer, Block
from .contrastive import ContrastivePath
from .utils import get_embedding_size_from_arch, get_masked_captions, get_att_mask
from Harmony.utils import get_2d_sincos_pos_embed
from Harmony.losses import MaskeDistLoss
from Harmony.models.heads import DINOHead, iBOTHead

class MaskCLIP(torch.nn.Module):
    def __init__(self, args, meta_training_data=None):
        super().__init__()
        self.meta = vars(args)
        if meta_training_data != None:
            self.meta = {**self.meta, **meta_training_data}
        self.meta["use_siglip"] = False 
        
        # define the model arch 
        self.define_arch() 

    def define_arch(self):
        # define the image encoder(s)
        try: 
            self.image_model = vits.__dict__[self.meta['arch']](
                patch_size=self.meta['patch_size'],
                drop_path_rate=self.meta['drop_path_rate'],
                can_be_contrastive=True,
            )
        except:
            raise Exception(f"Unknow arch: {self.meta['arch']}")
        self.student = self.image_model
        self.meta['embed_dim'] = self.student.embed_dim
        
        # define the decoder
        self.define_decoder(decoder_embed_dim=self.meta['embed_dim'], decoder_depth=1, decoder_num_heads=16)


        print("Defining teacher encoder")
        self.teacher = vits.__dict__[self.meta['arch']](
            patch_size=self.meta['patch_size'],
            can_be_contrastive=True,
        )
        if utils.has_batchnorms(self.student):
            self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
            self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)
        
        if self.meta["with_head"]:
            print("Creating iBOT heads")
            self.student_head = iBOTHead(
                self.meta['embed_dim'], 
                8192,
                patch_out_dim=8192,
                norm=None,
                act='gelu',
                norm_last_layer=True,
                shared_head=True ,
            )
            self.teacher_head = iBOTHead(
                self.meta['embed_dim'], 
                8192,
                patch_out_dim=8192,
                norm=None,
                act='gelu',
                shared_head=True,
            )
            self.student = HEAD_WRAPPER(self.student, self.student_head)
            self.teacher = HEAD_WRAPPER(self.teacher, self.teacher_head)

        
        loss_embed = self.meta['embed_dim']
        if self.meta["with_head"]:
            loss_embed = 8192
        
        self.loss = MaskeDistLoss(               
                loss_embed,
                loss_embed,
                self.meta['warmup_teacher_temp'],
                self.meta['teacher_temp'],
                self.meta['warmup_teacher_patch_temp'],
                self.meta['teacher_patch_temp'],
                self.meta['warmup_teacher_temp_epochs'],
                self.meta['epochs'],
                lambda1=self.meta['lambda1'],
                lambda2=self.meta['lambda2'],
                with_cls=self.meta['with_cls']
                )
        
        self.teacher.load_state_dict(self.student.state_dict(), strict=False)        
        self.contrastive_path = ContrastivePath(image_backbone=self.image_model, meta=self.meta)
        self.text_student = self.contrastive_path.text_backbone
        print("Using masked language modeling")
        self.mlm_head = TextDecoder(transformer_layers=4)

    def define_decoder(self,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4, norm_layer=nn.LayerNorm):
        self.patch_embed = self.student.patch_embed
        num_patches = self.student.patch_embed.num_patches

        # self.decoder_embed = nn.Linear(self.meta['embed_dim'], decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

    def initialize_deocder_weights(self):
    
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.image_model.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # add pos embed w/o cls token
        x = x + self.image_model.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.image_model.cls_token + self.image_model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.image_model.blocks:
            x = blk(x)
        x = self.image_model.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        # x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        return x
            
    def forward(self, images, epoch, captions=None):
        loss = torch.tensor([0.0]).to(self.meta['gpu'])
        outputs = {'loss': loss}

        # get masked embd from student, and teacher embed
        if self.meta["with_head"]:
            student_embd, mask, ids_restore = self.forward_encoder(images, mask_ratio=0.75)
            student_embd = self.forward_decoder(student_embd, ids_restore=ids_restore)
            x1, x2 = self.student_head(student_embd)
            x1 = x1.unsqueeze(1)
            student_embd = torch.cat((x1, x2), dim=1)

            teacher_embd = self.teacher.backbone(images, return_all_tokens=True)
            x1, x2 = self.teacher_head(teacher_embd)
            x1 = x1.unsqueeze(1)
            teacher_embd = torch.cat((x1, x2), dim=1)
        else:
            student_embd, mask, ids_restore = self.forward_encoder(images, mask_ratio=0.75)
            student_embd = self.forward_decoder(student_embd, ids_restore=ids_restore)
            teacher_embd = self.teacher(images, return_all_tokens=True)

        # mask self distillation loss
        mask_loss = self.loss(student_embd, teacher_embd, mask, epoch)
        outputs['mask_dist_loss'] = mask_loss.item() * self.meta['mask_dist_weight']
        outputs['loss'] += mask_loss * self.meta['mask_dist_weight']

        # clip loss
        output = self.contrastive_path.forward_(images, captions) 
        outputs['clip_loss'] = output['clip_loss'].item()
        outputs['loss'] += output['clip_loss']

        # mlm loss
        labels = captions.detach().clone()
        masked_captions, labels, masks_c = get_masked_captions(captions=captions, labels=labels)
        _, text_embedding = self.contrastive_path.text_backbone(masked_captions, return_without_proj=True)
        
        mlm_output = self.mlm_head(text_embedding)
        
        probs = mlm_output.view(-1, mlm_output.size(-1)) 
        labels = labels.view(-1)
        mlm_loss = torch.nn.functional.cross_entropy(probs, labels)
        if torch.isnan(mlm_loss):
            mlm_loss = torch.tensor(0.0) 
        outputs['mlm_loss'] = mlm_loss.item() * self.meta["mlm_weight"]
        outputs['loss'] += mlm_loss * self.meta["mlm_weight"]

        return outputs
    

class HEAD_WRAPPER(torch.nn.Module):
    def __init__(self, s, h):
        super().__init__()
        self.backbone = s
        self.head = h
        
# class HEAD_WRAPPER(torch.nn.Module):
#     def __init__(self, s, h, i = False):
#         super().__init__()
#         self.backbone = s
#         self.head = h
#         self.is_teacher = i
        

#     def forward(self, x):
#         if self.is_teacher:
#             x = self.backbone(x, return_all_tokens=True)
#             x1, x2 = self.head(x) 
#             x = torch.cat((x1, x2), dim=1)
#             return x
#         else:
#             x, mask, ids_restore = self.backbone.forward_encoder(x, mask_ratio=0.75)
#             x = self.forward_decoder(x, ids_restore=ids_restore)
#             x1, x2 = self.head(x)
#             x = torch.cat((x1, x2), dim=1)
#             return x, mask
            