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
    
class CustomTextHeadSequential(nn.Module):
    def __init__(self, text_backbone, distillation_head):
        super(CustomTextHeadSequential, self).__init__()
        self.text_backbone = text_backbone
        self.distillation_head = distillation_head

    def forward(self, x):
        _, x = self.text_backbone(x, return_without_proj=True)
        x = self.distillation_head(x)
        return x
    
    def only_head(self, x):
        return self.distillation_head(x)