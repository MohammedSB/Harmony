import torch
import torch.nn as nn
import torch.nn.functional as F

import Harmony.utils as utils

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, image_embed, text_embed, logit_scale, image_embed_teacher=None,
                text_embed_teacher=None, hard_weight=1.0, temp=0.1, logit_bias=None):
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        image_embed_all, text_embed_all = \
            utils.all_gather_batch_with_grad([image_embed, text_embed])

        if logit_bias: # proxy for using siglip
            global_batch_size = image_embed_all.shape[0]
            logits = (logit_scale * image_embed_all @ text_embed_all.t()) + logit_bias
            targets = 2 * torch.eye(global_batch_size) - torch.ones((global_batch_size, global_batch_size)) 

            targets = targets.to(device=image_embed_all.device)

            loss = -F.logsigmoid(targets * logits).sum() / global_batch_size
            loss = hard_weight * loss
        else:
            # cosine similarity as logits
            logits_per_image = logit_scale * image_embed @ text_embed_all.t()
            logits_per_text = logit_scale * text_embed @ image_embed_all.t()

            image_loss = F.cross_entropy(logits_per_image, self.labels)
            text_loss = F.cross_entropy(logits_per_text, self.labels)
            loss = hard_weight * ((image_loss + text_loss) / 2)

        return_dict = {}

        if hard_weight != 1.0 and image_embed_teacher != None:

            # normalized features
            image_embed_teacher = F.normalize(image_embed_teacher, dim=-1, p=2)
            text_embed_teacher = F.normalize(text_embed_teacher, dim=-1, p=2)

            image_embed_teacher_all, text_embed_teacher_all = utils.all_gather_batch_with_grad([image_embed_teacher, text_embed_teacher])

            targets_per_image_teacher =  F.softmax((image_embed_teacher @ text_embed_teacher_all.t())/temp, dim=1)
            targets_per_text_teacher =  F.softmax((text_embed_teacher @ image_embed_teacher_all.t())/temp, dim=1)

            image_loss_teacher = F.cross_entropy(logits_per_image, targets_per_image_teacher) 
            text_loss_teacher = F.cross_entropy(logits_per_text, targets_per_text_teacher) 

            soft_weight = 1.0 - hard_weight
            soft_loss =  (image_loss_teacher + text_loss_teacher) / 2 
            soft_loss_scaled = soft_weight * soft_loss
            loss += soft_loss_scaled
            return_dict['soft_loss'] = soft_loss

        return_dict['clip_loss'] = loss 
        return return_dict