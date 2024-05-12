import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

class MaskeDistLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0, mim_start_epoch=0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        )) 

    def forward(self, student_output, teacher_output, mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output[:, 0], student_output[:, 1:] 
        teacher_cls, teacher_patch = teacher_output[:, 0], teacher_output[:, 1:] 

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_patch = student_patch / self.student_temp
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach()
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach()

        # CLS level loss
        cls_loss = torch.sum(-teacher_cls_c * F.log_softmax(student_cls, dim=-1), dim=-1)
        cls_loss = cls_loss.mean()

        # Patch level loss
        patch_loss = torch.sum(-teacher_patch_c * F.log_softmax(student_patch, dim=-1), dim=-1)
        patch_loss = torch.sum(patch_loss * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        patch_loss = patch_loss.mean()

        loss = patch_loss + cls_loss

        self.update_center(teacher_cls, teacher_patch)         
        return loss                      

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)