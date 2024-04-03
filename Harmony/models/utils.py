import torch
from torch import nn

def get_embedding_size_from_arch(arch):
    if arch == "vit_tiny":
        return 192
    elif arch == "vit_small":
        return 384
    elif arch == "vit_base":
        return 768
    elif arch == "vit_large":
        return 1024

def get_masked_captions(captions, labels):
    masks_c = torch.ones_like(captions) * 0.20
    masks_c = torch.bernoulli(masks_c)

    # zero out all values past the eot.
    argmax_indices = captions.argmax(dim=1)
    range_tensor = torch.arange(captions.shape[1], device=captions.device).expand_as(captions)
    condition_mask = range_tensor > argmax_indices.unsqueeze(1)
    condition_mask[:, 0:1] = 1 # make all sot zero. 1 means add it for condition to remove. 
    row_indices = torch.arange(condition_mask.size(0))
    condition_mask[row_indices, argmax_indices] = 1 # make all eot zero.
    masks_c[condition_mask] = 0 # make sot and eot zero.

    # get masked input captions
    masked_captions = captions.clone()
    masked_captions[masks_c == 1] = 49408 # This is mask ID. TODO: do not hard code this                
    labels[masks_c == 0] = -100 # this is the default ignore value for pytorch ce

    return masked_captions, labels, masks_c


class CustomTextHeadSequential(nn.Module):
    def __init__(self, text_backbone, distillation_head):
        super(CustomTextHeadSequential, self).__init__()
        self.backbone = text_backbone
        self.distillation_head = distillation_head

    def forward(self, x):
        _, x = self.backbone(x, return_without_proj=True)
        x = self.distillation_head(x)
        return x
    
    def only_head(self, x):
        return self.distillation_head(x)
