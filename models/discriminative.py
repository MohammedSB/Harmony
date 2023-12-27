import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from vision_transformer import Block

import vision_transformer as vits
from vision_transformer import DINOHead
import utils

from losses import DINOLoss

class DiscriminativePath(nn.Module):
    def __init__(self, image_encoder, meta):
        super().__init__()
        self.image_encoder = image_encoder
        self.meta = meta

        # ============ building student and teacher networks ... ============
        self.teacher = vits.__dict__[self.meta['arch']](patch_size=self.meta['patch_size'])

        # multi-crop wrapper handles forward with inputs of different resolutions
        self.student_head = DINOHead(
            self.meta['embed_dim'],
            self.meta['out_dim'],
            use_bn=self.meta['use_bn_in_head'],
            norm_last_layer=self.meta['norm_last_layer'],
        )
        self.student = utils.MultiCropWrapper(self.image_encoder, self.student_head)
        self.teacher = utils.MultiCropWrapper(
            self.teacher,
            DINOHead(self.meta['embed_dim'], self.meta['out_dim'], self.meta['use_bn_in_head']),
        )
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
        self.teacher_without_ddp.load_state_dict(self.student.module.state_dict())
        # there is no backpropagation through the self.teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        print(f"Student and Teacher are built: they are both {self.meta['arch']} network.")

        if "dino" in self.meta['objective']:
            # ============ preparing loss ... ============
            self.loss = DINOLoss(
                self.meta['out_dim'],
                self.meta['local_crops_number'] + 2,  # total number of crops = 2 global crops + local_crops_number
                self.meta['warmup_teacher_temp'],
                self.meta['teacher_temp'],
                self.meta['warmup_teacher_temp_epochs'],
                self.meta['epochs'],
            ).cuda()

    def forward(self, images, epoch):
        teacher_output = self.teacher(images[:2])  # only the 2 global views pass through the teacher
        student_output = self.student(images)

        loss = self.loss(student_output, teacher_output, epoch)

        return {
            "teacher_output": teacher_output,
            "student_output": student_output,
            "loss": loss
        }