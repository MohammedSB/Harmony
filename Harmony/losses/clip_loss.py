import torch
import torch.nn as nn
import torch.nn.functional as F

import Harmony.utils as utils

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, image_embed, text_embed, logit_scale, logit_scale_teacher=None, image_embed_teacher=None, text_embed_teacher=None, hard_weight=1.0, temp=0.1):
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # # gather features from all GPUs
        image_embed_all, text_embed_all = \
            utils.all_gather_batch([image_embed, text_embed])
        
        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        image_loss = F.cross_entropy(logits_per_image, self.labels)
        text_loss = F.cross_entropy(logits_per_text, self.labels)

        loss = hard_weight * ((image_loss + text_loss) / 2)

        if hard_weight != 1.0 and image_embed_teacher != None:

            # normalized features
            image_embed_teacher = F.normalize(image_embed_teacher, dim=-1, p=2)
            text_embed_teacher = F.normalize(text_embed_teacher, dim=-1, p=2)

            image_embed_teacher_all, text_embed_teacher_all = utils.all_gather_batch([image_embed_teacher, text_embed_teacher])

            logits_per_image_teacher =  (image_embed_teacher @ text_embed_teacher_all.t())/temp
            logits_per_text_teacher = (text_embed_teacher @ image_embed_teacher_all.t())/temp 

            image_loss_teacher = F.cross_entropy(logits_per_image, logits_per_image_teacher) 
            text_loss_teacher = F.cross_entropy(logits_per_text, logits_per_text_teacher) 

            soft_weight = 1.0 - hard_weight

            loss = loss + (soft_weight * ((image_loss_teacher + text_loss_teacher) / 2 )) # add scaled soft loss

        return {'clip_loss': loss}