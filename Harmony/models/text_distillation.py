import torch
from Harmony.models.text_encoder import TextEncoder
from Harmony.models.heads.text_dist_head import TextDistillationHead 
from Harmony.models.utils import CustomTextHeadSequential
from Harmony.losses.text_dist_loss import TextDistillationLoss

class TextDistillationPath(torch.nn.Module):
    def __init__(self, meta, text_student, text_teacher=None, text_embed_dim = 512):
        super().__init__()

        self.meta = meta
        text_dist_out_dim = self.meta['out_dim'] // 4
        self.text_dist_student = CustomTextHeadSequential(text_student, TextDistillationHead(
            text_embed_dim, # this is the out_dim for text encoder
            text_dist_out_dim,
            norm=self.meta['norm_in_head'],
            norm_last_layer=self.meta['norm_last_layer']
        ))
        if text_teacher !=None : # check if we have a teacher backbone from contrastive path
            print("Using teacher backbone from contrastive path in text self-dist")
            text_dist_teacher = text_teacher
        else:
            print("Defininig a new teacher backbone for text self-dist")
            vocab_size = 49408 + 1
            text_dist_teacher = TextEncoder(embed_dim=text_embed_dim, vocab_size=vocab_size)

        self.text_dist_teacher = CustomTextHeadSequential(text_dist_teacher, TextDistillationHead(
            text_embed_dim,
            text_dist_out_dim,
            norm=self.meta['norm_in_head'],
            norm_last_layer=self.meta['norm_last_layer']
        ))
        for param in self.text_dist_teacher.parameters():
            param.requires_grad = False

        self.text_distillation_loss = TextDistillationLoss(
            text_embed_dim,
            text_dist_out_dim,
            self.meta['warmup_teacher_patch_temp'],
            self.meta['teacher_patch_temp'],
            self.meta['warmup_teacher_temp_epochs'],
            self.meta['epochs']
        )

    def forward(self, captions, masks_c, epoch, text_embedding=None):
        if text_embedding != None:
            student_text_embeddings = self.text_dist_student.only_head(text_embedding)
        else:
            student_text_embeddings = self.text_dist_student(text_embedding)
        teacher_text_embeddings = self.text_dist_teacher(captions)

        loss = self.text_distillation_loss(student_text_embeddings, teacher_text_embeddings,
                                            masks_c, epoch)
        if torch.isnan(loss):
            loss = torch.tensor(0.0) 

        return loss
        