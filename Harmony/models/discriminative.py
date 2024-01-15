import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from Harmony.models.vision_transformer import Block

import Harmony.models.vision_transformer as vits
from Harmony.models.heads import DINOHead, iBOTHead
from Harmony import utils
from Harmony.losses import DINOLoss, iBOTLoss, KoLeoLoss

class DiscriminativePath(nn.Module):
    def __init__(self, image_encoder, meta):
        super().__init__()
        self.image_encoder = image_encoder
        self.meta = meta

        # ============ building student and teacher networks ... ============
        self.teacher = vits.__dict__[self.meta['arch']](
            patch_size=self.meta['patch_size'],
            return_all_tokens=True if "ibot" in self.meta['objective'] else False,
            )

        # multi-crop wrapper handles forward with inputs of different resolutions
        if "dino" in self.meta['objective']:
            self.student_head = DINOHead(
                self.meta['embed_dim'],
                self.meta['out_dim'],
                norm=self.meta['norm_in_head'],
                norm_last_layer=self.meta['norm_last_layer'],
            )
            self.teacher_head = DINOHead(self.meta['embed_dim'], self.meta['out_dim'], self.meta['norm_in_head'])
        elif "ibot" in self.meta['objective']:
            self.student_head = iBOTHead(
                self.meta['embed_dim'], 
                self.meta['out_dim'],
                patch_out_dim=self.meta['patch_out_dim'],
                norm=self.meta['norm_in_head'],
                act=self.meta['act_in_head'],
                norm_last_layer=self.meta['norm_last_layer'],
                shared_head=self.meta['shared_head'],
            )
            self.teacher_head = iBOTHead(
                self.meta['embed_dim'], 
                self.meta['out_dim'],
                patch_out_dim=self.meta['patch_out_dim'],
                norm=self.meta['norm_in_head'],
                act=self.meta['act_in_head'],
                shared_head=self.meta['shared_head_teacher'],
            )

        self.student = utils.MultiCropWrapper(self.image_encoder, self.student_head)
        self.teacher = utils.MultiCropWrapper(self.teacher, self.teacher_head)

        # move networks to gpu
        self.student, self.teacher = self.student.cuda(), self.teacher.cuda()

        # synchronize batch norms (if any)
        if utils.has_batchnorms(self.student):
            self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
            self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)

            # we need DDP wrapper to have synchro batch norms working...
            self.teacher = nn.parallel.DistributedDataParallel(self.teacher, device_ids=[self.meta['gpu']])
            self.teacher_without_ddp = self.teacher.module
        else:
            # teacher_without_ddp and self.teacher are the same thing
            self.teacher_without_ddp = self.teacher
        self.student = nn.parallel.DistributedDataParallel(self.student, device_ids=[self.meta['gpu']])
        # self.teacher and self.student start with the same weights
        self.teacher_without_ddp.load_state_dict(self.student.module.state_dict(), strict=False)
        # there is no backpropagation through the self.teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        print(f"Student and Teacher are built: they are both {self.meta['arch']} network.")

        # ============ preparing loss ... ============
        if "dino" in self.meta['objective']:
            self.loss = DINOLoss(
                self.meta['out_dim'],
                self.meta['local_crops_number'] + 2,  # total number of crops = 2 global crops + local_crops_number
                self.meta['warmup_teacher_temp'],
                self.meta['teacher_temp'],
                self.meta['warmup_teacher_temp_epochs'],
                self.meta['epochs'],
            ).cuda()
        elif "ibot" in self.meta['objective']:
            same_dim = self.meta['shared_head'] or self.meta['shared_head_teacher']
            self.loss = iBOTLoss(
                self.meta['out_dim'],
                self.meta['out_dim'] if same_dim else self.meta['patch_out_dim'],
                self.meta['global_crops_number'],
                self.meta['local_crops_number'],
                self.meta['warmup_teacher_temp'],
                self.meta['teacher_temp'],
                self.meta['warmup_teacher_patch_temp'],
                self.meta['teacher_patch_temp'],
                self.meta['warmup_teacher_temp_epochs'],
                self.meta['epochs'],
                lambda1=self.meta['lambda1'],
                lambda2=self.meta['lambda2'],
                mim_start_epoch=self.meta['pred_start_epoch'],
            ).cuda()

    def forward(self, images, epoch, masks):
        
        teacher_output = self.teacher(images[:self.meta['global_crops_number']])

        if 'dino' in self.meta['objective']:
            student_output = self.student(images)

            loss = self.loss(student_output, teacher_output, epoch)
        elif 'ibot' in self.meta['objective']:

            backbone_feat, student_output = self.student(images[:self.meta['global_crops_number']], mask=masks[:self.meta['global_crops_number']], return_backbone_feat=True)

            # print(len(backbone_feat[:, 0]))
            # print(backbone_feat[0].shape)

            # get local views
            self.student.module.backbone.masked_im_modeling = False
            student_local_cls = self.student(images[self.meta['global_crops_number']:])[0] if len(images) > self.meta['global_crops_number'] else None
            self.student.module.backbone.masked_im_modeling = self.meta['use_masked_im_modeling']

            all_loss = self.loss(student_output, teacher_output, student_local_cls, masks, epoch)

            loss = all_loss.pop('loss')

        # if self.use_koleo:
        #     self.koleo_loss = KoLeoLoss()
        #     koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
        #         self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
        #     )  # we don't apply koleo loss between cls tokens of a same image
        #     loss_accumulator += koleo_loss
        #     loss_dict["koleo_loss"] = (
        #         koleo_loss / loss_scales
        #         )  # this is to display the same losses as before but we can remove eventually

        return {
            "teacher_output": teacher_output,
            "student_output": student_output,
            "loss": loss
        }