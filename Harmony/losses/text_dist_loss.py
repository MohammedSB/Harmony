import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

class TextDistillationLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, 
                 warmup_teacher_temp, teacher_temp, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9,
                 mim_start_epoch=0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp,
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        student_output = student_output / self.student_temp
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]

        teacher_output = F.softmax((teacher_output - self.center) / temp, dim=-1)
        loss = torch.sum(-teacher_output * F.log_softmax(student_output, dim=-1), dim=-1)
        loss = torch.sum(loss * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        loss = loss.mean()

        self.update_center(teacher_output)                               
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        patch_center = torch.sum(teacher_output.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_output) * dist.get_world_size())
        self.center = self.center * self.center_momentum + patch_center * (1 - self.center_momentum)